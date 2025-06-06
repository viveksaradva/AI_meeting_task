import os
import umap
import json
import hdbscan
# from sklearn.cluster import DBSCAN
import logging
import requests
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
# from sentence_transformers import SentenceTransformer
from modules.db.postgres import retrieve_transcript
from modules.prompts import (
    llm_powered_disfluency_cleaner_prompt, llm_augmented_segment_validator_prompt
)

load_dotenv()
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

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

    def clean(self, text: str) -> str:
        response = self.chain.invoke({"disfluenced_text": text})
        return response.content

class SegmentValidator:
    def __init__(self):
        self.chain = llm_augmented_segment_validator_prompt | chat

    @sleep_and_retry
    @limits(calls=10, period=60)
    def validate(self, prev_context: str, curr_context: str) -> float:
        response = self.chain.invoke({
            "prev_context": prev_context,
            "curr_context": curr_context
        })
        parsed = json.loads(response.content)
        return float(parsed.get("confidence", 0))

##############################################################
# The SegmenterAlgorithm class for segmenting the transcript #
##############################################################
class SegmenterAlgorithm:
    def __init__(self, embedding_model="nomic-embed-text", llm_validator_model="mistral-saba-24b"):
        self.transcript: List[Dict[str, str]] = []
        self.embedding_model = embedding_model
        self.llm_validator_model = llm_validator_model
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

    def reduce_dimensions(self, n_neighbors=15, min_dist=0.0, n_components=15, metric='cosine'):
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
    # def cluster_transcript(self, eps=0.8, min_samples=3):
    #     """
    #     Cluster the transcript using DBSCAN with adaptive parameters.
    #     """
    #     reduced_embeddings = self.reduce_dimensions()

    #     # Adaptive min_samples based on transcript length
    #     if min_samples is None:
    #         min_samples = max(3, len(self.transcript) // 20)

    #     clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    #     self.labels = clusterer.fit_predict(reduced_embeddings)
    #     logging.info(f"Identified {len(set(self.labels)) - (1 if -1 in self.labels else 0)} clusters (excluding noise).")
    #     return self.labels


    def build_segments_from_labels(self) -> List[Dict]:
        if self.labels is None:
            raise ValueError("No cluster labels. Run cluster_transcript() first.")
        if len(self.labels) != len(self.transcript):
            raise ValueError("Transcript/labels length mismatch.")

        # Group by label, track original index
        clusters: Dict[int, List[Dict]] = {}
        noise_buffer = []
        for idx, (entry, lbl) in enumerate(zip(self.transcript, self.labels)):
            entry["_index"] = idx
            if lbl == -1:
                noise_buffer.append(entry)
            else:
                clusters.setdefault(lbl, []).append(entry)
                # flush any buffered noise into this cluster
                if noise_buffer:
                    clusters[lbl].extend(noise_buffer)
                    noise_buffer = []

        # any trailing noise → last cluster
        if noise_buffer and clusters:
            last_lbl = sorted(clusters.keys())[-1]
            clusters[last_lbl].extend(noise_buffer)

        # Sort clusters by earliest utterance index
        ordered = sorted(
            clusters.items(),
            key=lambda kv: min(u["_index"] for u in kv[1])
        )

        # Emit segments list
        segments = []
        for lbl, utts in ordered:
            # Remove helper index
            for u in utts:
                u.pop("_index", None)
            # Sort by start timestamp
            utts = sorted(utts, key=lambda u: u["start"])
            segments.append({"label": lbl, "utterances": utts})
        return segments
    
    def validate_segments_with_llm(self, segments: List[Dict], percentile: float = 75.0) -> List[Dict]:
        if len(segments) < 2:
            return segments

        validator = SegmentValidator()
        boundary_confs = []
        for prev, curr in zip(segments, segments[1:]):
            ctx_prev = " ".join(u["utterance"] for u in prev["utterances"][-3:])
            ctx_curr = " ".join(u["utterance"] for u in curr["utterances"][:3])
            conf = validator.validate(ctx_prev, ctx_curr)
            boundary_confs.append(conf)

        threshold = float(np.percentile(boundary_confs, percentile))

        # Perform segment splitting based on confidence
        validated_segments = []
        curr_segment = {
            "label": 0,
            "utterances": segments[0]["utterances"]
        }

        for idx in range(1, len(segments)):
            confidence = boundary_confs[idx - 1]
            next_segment = segments[idx]

            if confidence >= threshold:
                # Boundary is valid: split into a new segment
                validated_segments.append(curr_segment)
                curr_segment = {
                    "label": len(validated_segments),
                    "utterances": next_segment["utterances"]
                }
            else:
                # Boundary is weak: merge next utterances into current segment
                curr_segment["utterances"].extend(next_segment["utterances"])

        # Append the last segment
        validated_segments.append(curr_segment)

        # Annotate confidence only for true splits
        for i, conf in enumerate(boundary_confs):
            if conf >= threshold and i + 1 < len(validated_segments):
                validated_segments[i + 1]["boundary_confidence"] = conf

        return validated_segments
