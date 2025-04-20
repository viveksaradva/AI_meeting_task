import os
import umap
import json
import hdbscan
import logging
import requests
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
from modules.db.postgres import retrieve_transcript
from modules.prompts import llm_powered_disfluency_cleaner_prompt, llm_augmented_segment_validator_prompt

load_dotenv()

#############################
# For cleaning disfluencies #
#############################
chat = ChatGroq(
    temperature = 0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.1-8b-instant"
)

class DisfluencyCleaner:
    def __init__(self):
        self.chain = llm_powered_disfluency_cleaner_prompt | chat

    @sleep_and_retry
    @limits(calls=10, period=60)  # 10 calls per minute (60 seconds)
    def clean(self, text: str) -> str:
        try:
            response = self.chain.invoke({"disfluenced_text": text})
            # Handle different response formats
            if hasattr(response, 'content'):
                return response.content
            else:
                return response
        except Exception as e:
            logging.error(f"Error in DisfluencyCleaner: {e}")
            # Return the original text on error
            return text

class SegmentValidator:
    def __init__(self):
        self.chain = llm_augmented_segment_validator_prompt | chat

    @sleep_and_retry
    @limits(calls=10, period=60)  # 10 calls per minute (60 seconds)
    def validate(self, prev_context: str, curr_context: str) -> float:
        try:
            response = self.chain.invoke({
                "prev_context": prev_context,
                "curr_context": curr_context
            })
            # Handle different response formats
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = response

            # Parse the JSON response
            parsed = json.loads(content)
            return float(parsed.get("confidence", 0))
        except Exception as e:
            logging.error(f"Error in SegmentValidator: {e}")
            # Return a default confidence of 0 on error
            return 0.0

##############################################################
# The SegmenterAlgorithm class for segmenting the transcript #
##############################################################
class SegmenterAlgorithm:
    def __init__(self, embedding_model="nomic-embed-text", llm_validator_model="mistral-saba-24b"):
        self.transcript: List[Dict[str, str]] = []
        self.embedding_model = embedding_model
        self.llm_validator_model = llm_validator_model
        self.disfluency_classifier = pipeline("token-classification",
                                              model="4i-ai/BERT_disfluency_cls",
                                              aggregation_strategy="simple")
        self.disfluency_cleaner = DisfluencyCleaner()
        self.embeddings = None
        self.labels = None

    def load_transcript(self, meeting_id: str = None):
        """
        Loads the transcript for the specified meeting ID.

        Retrieves the transcript from the database and sets it to the instance variable `self.transcript`.

        :param meeting_id: The ID of the meeting for which the transcript is to be retrieved.
        """
        logging.info(f"Retrieving transcript for meeting: {meeting_id}")
        self.transcript = retrieve_transcript(meeting_id)

    def preprocess_text(self, text: str) -> str:
        # Use only the LLM-based cleaner for disfluency removal
        try:
            llm_cleaned_text = self.disfluency_cleaner.clean(text)
            return llm_cleaned_text
        except Exception as e:
            logging.warning(f"LLM disfluency cleaning failed: {e}. Returning original text.")
            return text

    def embed_transcript(self) -> np.ndarray:
        # Add parallel processing for larger transcripts
        if not self.transcript:
            raise ValueError("Transcript is empty. Please load it first.")

        def embed_utterance(entry):
            preprocessed = self.preprocess_text(entry["utterance"])
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.embedding_model, "prompt": preprocessed}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama embedding failed: {response.status_code} - {response.text}")
            return response.json()["embedding"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            embeddings = list(executor.map(embed_utterance, self.transcript))

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

    def cluster_transcript(self, min_cluster_size=None):
        """
        Cluster the transcript using HDBSCAN with adaptive parameters.
        """
        reduced_embeddings = self.reduce_dimensions()

        # Adaptive min_cluster_size based on transcript length
        if min_cluster_size is None:
            min_cluster_size = max(3, len(self.transcript) // 20)

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
        if len(segments) < 2:
            return segments

        # Initialize the segment validator
        validator = SegmentValidator()

        validated_segments = [segments[0]]
        for i in range(1, len(segments)):
            prev_segment = validated_segments[-1]
            current_segment = segments[i]

            # Extract contexts: use up to 3 utterances from end of previous segment and beginning of the current segment
            context_prev = " ".join([utt["utterance"] for utt in prev_segment["utterances"][-3:]])
            context_curr = " ".join([utt["utterance"] for utt in current_segment["utterances"][:3]])

            try:
                # Use the SegmentValidator to get the confidence score
                confidence = validator.validate(context_prev, context_curr)
            except Exception as e:
                logging.error(f"Error calling LLM validator: {e}")
                # In case of failure, default to merging the segments to be conservative.
                validated_segments[-1]["utterances"].extend(current_segment["utterances"])
                continue

            logging.info(f"LLM confidence for boundary between segments {prev_segment['label']} and {current_segment['label']}: {confidence}")

            if confidence >= threshold:
                # Accept the boundary as a valid topic drift.
                validated_segments.append(current_segment)
            else:
                # Merge the segments if the topic drift is not strongly supported.
                validated_segments[-1]["utterances"].extend(current_segment["utterances"])

        # Add confidence scores to output
        for i, segment in enumerate(validated_segments):
            if i > 0:
                segment["boundary_confidence"] = confidence  # Store the confidence score

        return validated_segments