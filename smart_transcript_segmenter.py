import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple
import logging
from dotenv import load_dotenv
from modules.db.postgres import retrieve_transcript

load_dotenv()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class TranscriptSegmenter:
    def __init__(self):
        """
        Initializes the segmenter.
        """
        self.embedding_model = "nomic-embed-text"
        self.groq_model = "mistral-saba-24b"
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_api_key =  os.getenv("GROQ_API_KEY")

    def embed_utterances(self, transcript: List[Dict[str, any]]) -> np.ndarray:
        texts = [entry["utterance"] for entry in transcript]
        embeddings = []

        for text in texts:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama embedding request failed: {response.status_code} - {response.text}")

            data = response.json()
            embeddings.append(data["embedding"])

        return np.array(embeddings)

    def cluster_transcript(self, transcript: List[Dict[str, any]], k: int) -> Tuple[List[int], float]:
        embeddings = self.embed_utterances(transcript)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = kmeans.labels_
        sil_score = silhouette_score(embeddings, labels) if k > 1 else -1
        return labels.tolist(), sil_score

    def find_best_segmentation(self, transcript: List[Dict[str, any]], min_k: int = 2, max_k: int = 10) -> Dict:
        best_k, best_score, best_labels = min_k, -1, None
        for k in range(min_k, min(max_k, len(transcript)) + 1):
            labels, score = self.cluster_transcript(transcript, k)
            logging.info(f"k = {k}, Silhouette Score = {score:.3f}")
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        logging.info(f"Optimal number of clusters: {best_k} with Silhouette Score: {best_score:.3f}")
        return {"best_k": best_k, "labels": best_labels, "silhouette": best_score}

    def assign_segments(self, transcript: List[Dict[str, any]], labels: List[int]) -> List[List[Dict[str, any]]]:
        segments = {}
        for entry, label in zip(transcript, labels):
            segments.setdefault(label, []).append(entry)
        sorted_segments = [segments[label] for label in sorted(segments.keys())]
        return sorted_segments

    def plot_tsne(self, transcript: List[Dict[str, any]], labels: List[int]):
        embeddings = self.embed_utterances(transcript)
        n_samples = len(embeddings)
        perplexity = min(30, max(2, n_samples // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set2", s=60)
        plt.colorbar(label="Cluster Label")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.title("TSNE Visualization of Transcript Embeddings")
        plt.show()

    def annotate_segments_with_topics(self, segments: List[List[Dict[str, any]]]) -> List[Dict[str, any]]:
        """
        Annotate each segment with a topic using Groq-hosted LLM.
        """
        annotated = []
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        for idx, segment in enumerate(segments):
            text_block = "\n".join(f"{entry['speaker']}: {entry['utterance']}" for entry in segment)
            prompt = f"Summarize the main topic or theme of the following meeting discussion segment:\n\n{text_block}\n\nTopic:"
            response = requests.post(
                self.groq_api_url,
                headers=headers,
                json={
                    "model": self.groq_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that extracts concise topics from meeting segments."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"Groq topic labeling failed: {response.status_code} - {response.text}")
            result = response.json()
            topic = result["choices"][0]["message"]["content"]
            annotated.append({"topic": topic.strip(), "segment": segment})
        return annotated

if __name__ == "__main__":
    transcript = retrieve_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")

    segmenter = TranscriptSegmenter()
    scores = segmenter.find_best_segmentation(transcript, min_k=2, max_k=5)
    best_labels = scores["labels"]
    segments = segmenter.assign_segments(transcript, best_labels)
    annotated = segmenter.annotate_segments_with_topics(segments)

    print("Optimal Clustering:", scores)
    print("Annotated Transcript Segments:")
    for idx, annotated_seg in enumerate(annotated):
        print(f"\nSegment {idx} - Topic: {annotated_seg['topic']}")
        for entry in annotated_seg["segment"]:
            print(f"  {entry['speaker']}: {entry['utterance']}")

    segmenter.plot_tsne(transcript, best_labels)
