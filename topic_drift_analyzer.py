import logging
import requests
import umap
import json
import torch
import hdbscan  
import numpy as np
from typing import List, Dict
from modules.db.postgres import retrieve_transcript
from modules.llm import get_groq_response
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

#############################
# For cleaning disfluencies #
#############################
class DisfluencyCleaner:
    def __init__(self, model_name="4i-ai/BERT_disfluency_cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()

    def clean(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

        # Remove tokens classified as filler (label == 1)
        cleaned_tokens = [
            token for token, pred in zip(tokens, predictions)
            if pred == 0 and not token.startswith("##")
        ]

        cleaned_text = self.tokenizer.convert_tokens_to_string(cleaned_tokens)
        return cleaned_text

##############################################################
# The SegmenterAlgorithm class for segmenting the transcript #
##############################################################
class SegmenterAlgorithm:
    def __init__(self):
        self.transcript: List[Dict[str, str]] = []
        self.embedding_model = "nomic-embed-text"
        self.llm_validator_model = "mistral-saba-24b"
        self.disfluency_classifier = pipeline("token-classification",
                                              model="4i-ai/BERT_disfluency_cls",
                                              aggregation_strategy="simple")
        self.embeddings = None
        self.labels = None

    def load_transcript(self, meeting_id: str = None):
        if meeting_id:
            logging.info(f"Retrieving transcript for meeting: {meeting_id}")
            self.transcript = retrieve_transcript(meeting_id)
        else:
            logging.info("Using fallback sample transcript.")
            self.transcript = [
                {"speaker": "Speaker_0", "utterance": "Hello everyone, let's begin by discussing our roadmap."},
                {"speaker": "Speaker_1", "utterance": "Sure, I have some updates on the product launch and market strategy."},
                {"speaker": "Speaker_0", "utterance": "Great, let's also review our timelines and dependencies."},
                {"speaker": "Speaker_1", "utterance": "I am concerned about the delays in the supply chain affecting our launch."},
                {"speaker": "Speaker_0", "utterance": "Understood, we'll need to address that in the next meeting."},
                {"speaker": "Speaker_1", "utterance": "Also, we should consider feedback from our initial customers."},
                {"speaker": "Speaker_0", "utterance": "Exactly, customer feedback is essential to refine our approach."},
                {"speaker": "Speaker_1", "utterance": "Let's schedule a follow-up with the customer relations team."}
            ]
    def preprocess_text(self, text: str) -> str:
        results = self.disfluency_classifier(text)
        char_mask = [True] * len(text)
        for entity in results:
            if "DIS" in entity.get("entity", "").upper():
                for i in range(entity["start"], entity["end"]):
                    if i < len(char_mask):
                        char_mask[i] = False
        cleaned_text = "".join([ch for ch, keep in zip(text, char_mask) if keep])
        return " ".join(cleaned_text.split())

    def embed_transcript(self) -> np.ndarray:
        if not self.transcript:
            raise ValueError("Transcript is empty. Please load it first.")

        embeddings = []
        for entry in self.transcript:
            preprocessed = self.preprocess_text(entry["utterance"])
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.embedding_model, "prompt": preprocessed}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama embedding failed: {response.status_code} - {response.text}")
            data = response.json()
            embeddings.append(data["embedding"])

        self.embeddings = np.array(embeddings)
        return self.embeddings

    def reduce_dimensions(self, n_neighbors=15, min_dist=0.0, n_components=5, metric='cosine'):
        """
        Reduce dimensionality of embeddings using UMAP.
        """
        n_neighbors = min(15, len(self.embeddings) - 1)
        if self.embeddings is None:
            raise ValueError("Embeddings not found. Please run embed_transcript() first.")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            n_components=n_components, metric=metric)
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        return reduced_embeddings

    def cluster_transcript(self, min_cluster_size=5):
        """
        Cluster the transcript using HDBSCAN.
        """
        reduced_embeddings = self.reduce_dimensions()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        self.labels = clusterer.fit_predict(reduced_embeddings)
        logging.info(f"Identified {len(set(self.labels)) - (1 if -1 in self.labels else 0)} clusters (excluding noise).")
        return self.labels

    def build_segments_from_labels(self) -> List[Dict]:
        """
        Groups utterances into segments based on HDBSCAN cluster labels.
        Returns a list of segments where each segment contains utterances from the same cluster.
        """
        if self.labels is None:
            raise ValueError("No cluster labels found. Please run cluster_transcript() first.")
        
        if len(self.transcript) != len(self.labels):
            raise ValueError("Mismatch between transcript and label lengths.")

        segments = {}
        for entry, label in zip(self.transcript, self.labels):
            if label == -1:
                continue  # Skip noise
            segments.setdefault(label, []).append(entry)

        # Sort segments by the first utterance index to preserve rough order
        ordered_segments = sorted(segments.items(), key=lambda kv: self.labels.tolist().index(kv[0]))

        # Convert to list of dicts
        return [{"label": label, "utterances": utterances} for label, utterances in ordered_segments]

    def validate_segments_with_llm(self, segments: List[Dict], threshold: float = 7.0) -> List[Dict]:
        """
        For each consecutive segment boundary, validate with LLM if the topic drift is genuine.
        If the LLM's confidence (0 to 10) is below the threshold, merge the segments.
        """
        if len(segments) < 2:
            return segments

        validated_segments = [segments[0]]
        for i in range(1, len(segments)):
            prev_segment = validated_segments[-1]
            current_segment = segments[i]

            # Extract contexts: use up to 3 utterances from end of previous segment and beginning of the current segment
            context_prev = " ".join([utt["utterance"] for utt in prev_segment["utterances"][-3:]])
            context_curr = " ".join([utt["utterance"] for utt in current_segment["utterances"][:3]])
            prompt = (
                f"""Compare the following two transcript contexts and determine if there is a significant topic drift between them.

                Return your response in the following strict JSON format:

                {{
                "confidence": <a number between 0 and 10>,
                "explanation": "<a brief sentence explaining the level of topic drift>"
                }}
                Follow these rules:
                        1. NEVER use markdown code blocks (```json or ```)
                        2. ALWAYS return raw JSON without any wrapping characters
                        3. ALWAYS return valid JSON

                Context 1: {context_prev}

                Context 2: {context_curr}"""
            )

            try:
                llm_output = get_groq_response(prompt, self.llm_validator_model)
            except Exception as e:
                logging.error(f"Error calling LLM validator: {e}")
                # In case of failure, default to merging the segments to be conservative.
                validated_segments[-1]["utterances"].extend(current_segment["utterances"])
                continue

            # Parse LLM output to extract confidence
            # confidence = 0.0
            try:
                # Optional: If using structured JSON format, uncomment the following
                parsed = json.loads(llm_output)
                confidence = float(parsed.get("confidence", 0))

            except Exception:
                logging.warning(f"Failed to parse LLM response: {llm_output}")
                confidence = 0.0

            logging.info(f"LLM confidence for boundary between segments {prev_segment['label']} and {current_segment['label']}: {confidence}")

            if confidence >= threshold:
                # Accept the boundary as a valid topic drift.
                validated_segments.append(current_segment)
            else:
                # Merge the segments if the topic drift is not strongly supported.
                validated_segments[-1]["utterances"].extend(current_segment["utterances"])

        return validated_segments
 

if __name__ == "__main__":
    segmenter = SegmenterAlgorithm()
    segmenter.load_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")
    # segmenter.load_transcript()
    segmenter.embed_transcript()
    segmenter.cluster_transcript()
    segments = segmenter.build_segments_from_labels()
    
    # Optionally, print original segments before LLM validation.
    print("\n--- Original Segments ---")
    for idx, seg in enumerate(segments):
        print(f"\nSegment {idx} (Label {seg['label']}):")
        for utt in seg["utterances"]:
            print(f"{utt['speaker']}: {utt['utterance']}")
    
    # Apply LLM-based augmented validation to refine segment boundaries.
    validated_segments = segmenter.validate_segments_with_llm(segments, threshold=7.0)
    
    print("\n--- Validated Segments (After LLM Augmented Validation) ---")
    for idx, seg in enumerate(validated_segments):
        print(f"\nSegment {idx} (Label {seg.get('label', 'merged')}):")
        for utt in seg["utterances"]:
            print(f"{utt['speaker']}: {utt['utterance']}")
