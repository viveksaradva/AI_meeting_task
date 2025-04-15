import logging
import requests
import umap
import hdbscan  
import numpy as np
from typing import List, Dict
from modules.db.postgres import retrieve_transcript

class SegmenterAlgorithm:
    def __init__(self):
        self.transcript: List[Dict[str, str]] = []
        self.embedding_model = "nomic-embed-text"
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

    def embed_transcript(self) -> np.ndarray:
        if not self.transcript:
            raise ValueError("Transcript is empty. Please load it first.")

        embeddings = []
        for entry in self.transcript:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.embedding_model, "prompt": entry["utterance"]}
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

    def cluster_transcript(self, min_cluster_size=2):
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

if __name__ == "__main__":
    segmenter = SegmenterAlgorithm()
    # segmenter.load_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")
    segmenter.load_transcript()
    segmenter.embed_transcript()
    labels = segmenter.cluster_transcript()
    segments = segmenter.build_segments_from_labels()
    for idx, seg in enumerate(segments):
        print(f"\n--- Segment {idx} (Label {seg['label']}) ---")
        for utt in seg["utterances"]:
            print(f"{utt['speaker']}: {utt['utterance']}")
    # print(f"Labels: {labels}")
