# import numpy as np
# import requests
# import os
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from typing import List, Dict, Tuple
# import logging
# from dotenv import load_dotenv
# from modules.db.postgres import retrieve_transcript

# load_dotenv()
# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# class TranscriptSegmenter:
#     def __init__(self):
#         """
#         Initializes the segmenter.
#         """
#         self.embedding_model = "nomic-embed-text"
#         self.groq_model = "mistral-saba-24b"
#         self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
#         self.groq_api_key =  os.getenv("GROQ_API_KEY")

#     def embed_utterances(self, transcript: List[Dict[str, any]]) -> np.ndarray:
#         texts = [entry["utterance"] for entry in transcript]
#         embeddings = []

#         for text in texts:
#             response = requests.post(
#                 "http://localhost:11434/api/embeddings",
#                 json={"model": self.embedding_model, "prompt": text}
#             )
#             if response.status_code != 200:
#                 raise RuntimeError(f"Ollama embedding request failed: {response.status_code} - {response.text}")

#             data = response.json()
#             embeddings.append(data["embedding"])

#         return np.array(embeddings)

#     def cluster_transcript(self, transcript: List[Dict[str, any]], k: int) -> Tuple[List[int], float]:
#         embeddings = self.embed_utterances(transcript)
#         kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
#         labels = kmeans.labels_
#         sil_score = silhouette_score(embeddings, labels) if k > 1 else -1
#         return labels.tolist(), sil_score

#     def find_best_segmentation(self, transcript: List[Dict[str, any]], min_k: int = 2, max_k: int = 10) -> Dict:
#         best_k, best_score, best_labels = min_k, -1, None
#         for k in range(min_k, min(max_k, len(transcript)) + 1):
#             labels, score = self.cluster_transcript(transcript, k)
#             logging.info(f"k = {k}, Silhouette Score = {score:.3f}")
#             if score > best_score:
#                 best_score, best_k, best_labels = score, k, labels
#         logging.info(f"Optimal number of clusters: {best_k} with Silhouette Score: {best_score:.3f}")
#         return {"best_k": best_k, "labels": best_labels, "silhouette": best_score}

#     def assign_segments(self, transcript: List[Dict[str, any]], labels: List[int]) -> List[List[Dict[str, any]]]:
#         segments = {}
#         for entry, label in zip(transcript, labels):
#             segments.setdefault(label, []).append(entry)
#         sorted_segments = [segments[label] for label in sorted(segments.keys())]
#         return sorted_segments

#     def plot_tsne(self, transcript: List[Dict[str, any]], labels: List[int]):
#         embeddings = self.embed_utterances(transcript)
#         n_samples = len(embeddings)
#         perplexity = min(30, max(2, n_samples // 3))
#         tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#         reduced = tsne.fit_transform(embeddings)

#         plt.figure(figsize=(10, 6))
#         plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set2", s=60)
#         plt.colorbar(label="Cluster Label")
#         plt.xlabel("TSNE-1")
#         plt.ylabel("TSNE-2")
#         plt.title("TSNE Visualization of Transcript Embeddings")
#         plt.show()

#     def annotate_segments_with_topics(self, segments: List[List[Dict[str, any]]]) -> List[Dict[str, any]]:
#         """
#         Annotate each segment with a topic using Groq-hosted LLM.
#         """
#         annotated = []
#         headers = {
#             "Authorization": f"Bearer {self.groq_api_key}",
#             "Content-Type": "application/json"
#         }
#         for idx, segment in enumerate(segments):
#             text_block = "\n".join(f"{entry['speaker']}: {entry['utterance']}" for entry in segment)
#             prompt = f"Summarize the main topic or theme of the following meeting discussion segment:\n\n{text_block}\n\nTopic:"
#             response = requests.post(
#                 self.groq_api_url,
#                 headers=headers,
#                 json={
#                     "model": self.groq_model,
#                     "messages": [
#                         {"role": "system", "content": "You are a helpful assistant that extracts concise topics from meeting segments."},
#                         {"role": "user", "content": prompt}
#                     ]
#                 }
#             )
#             if response.status_code != 200:
#                 raise RuntimeError(f"Groq topic labeling failed: {response.status_code} - {response.text}")
#             result = response.json()
#             topic = result["choices"][0]["message"]["content"]
#             annotated.append({"topic": topic.strip(), "segment": segment})
#         return annotated

# if __name__ == "__main__":
#     transcript = retrieve_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")

#     segmenter = TranscriptSegmenter()
#     scores = segmenter.find_best_segmentation(transcript, min_k=2, max_k=5)
#     best_labels = scores["labels"]
#     segments = segmenter.assign_segments(transcript, best_labels)
#     annotated = segmenter.annotate_segments_with_topics(segments)

#     print("Optimal Clustering:", scores)
#     print("Annotated Transcript Segments:")
#     for idx, annotated_seg in enumerate(annotated):
#         print(f"\nSegment {idx} - Topic: {annotated_seg['topic']}")
#         for entry in annotated_seg["segment"]:
#             print(f"  {entry['speaker']}: {entry['utterance']}")

#     segmenter.plot_tsne(transcript, best_labels)
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging
from dotenv import load_dotenv
from modules.db.postgres import retrieve_transcript
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

# New library imports for improved clustering and visualization:
import hdbscan
import umap

load_dotenv()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class TranscriptSegmenter:
    def __init__(self):
        # Use a high-quality embedding model if available.
        self.embedding_model = "nomic-embed-text"
        self.groq_model = "mistral-saba-24b"
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_api_key = os.getenv("GROQ_API_KEY")
    
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
    
    def cluster_transcript(self, transcript: List[Dict[str, any]]) -> Tuple[List[int], Dict]:
        """
        Instead of using KMeans over a fixed range of clusters, this version uses HDBSCAN,
        which is capable of finding clusters of varying densities and also designates noise (-1).
        """
        embeddings = self.embed_utterances(transcript)
        # Parameters here can be tuned—for instance, min_cluster_size determines the smallest acceptable cluster.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True).fit(embeddings)
        labels = clusterer.labels_.tolist()
        # Report additional clustering statistics.
        cluster_info = {
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": labels.count(-1)
        }
        logging.info(f"HDBSCAN detected {cluster_info['n_clusters']} clusters with {cluster_info['n_noise']} noise points")
        return labels, cluster_info

    def assign_segments(self, transcript: List[Dict[str, any]], labels: List[int]) -> List[List[Dict[str, any]]]:
        # Group transcript entries by label; noise points (labeled as -1) will be handled separately.
        segments = {}
        for entry, label in zip(transcript, labels):
            # Optionally, if you want to treat noise as its own segment, you can leave it in.
            segments.setdefault(label, []).append(entry)
        sorted_segments = [segments[label] for label in sorted(segments.keys()) if label != -1]
        # If there are noise points and you prefer to attach them to the nearest segment,
        # you can implement custom logic here.
        return sorted_segments

    def refine_segments_with_llm(self, segments: List[List[Dict[str, any]]], max_iterations: int = 3) -> List[List[Dict[str, any]]]:
        """
        Iteratively refine segment boundaries using LLM augmentation.
        For each adjacent segment pair, the LLM is queried whether they share continuity.
        If so, they are merged. The process repeats for a number of iterations.
        """
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        iteration = 0
        while iteration < max_iterations:
            logging.info(f"--- Refinement Iteration {iteration + 1} ---")
            refined_segments = []
            idx = 0
            merge_occurred = False
            while idx < len(segments):
                if idx < len(segments) - 1:
                    left_segment = segments[idx]
                    right_segment = segments[idx + 1]
                    
                    # Use a slightly larger context window:
                    left_text = "\n".join(f"{e['speaker']}: {e['utterance']}" for e in left_segment[-3:])
                    right_text = "\n".join(f"{e['speaker']}: {e['utterance']}" for e in right_segment[:3])
                    
                    prompt = (
                        "You are a helpful assistant for meeting transcript segmentation.\n\n"
                        "Given the end of one segment and the beginning of the next, determine whether the two segments "
                        "should be merged because they continue the same topic. Respond with a single word: 'merge' or 'split'.\n\n"
                        f"--- Segment A ends with ---\n{left_text}\n\n"
                        f"--- Segment B begins with ---\n{right_text}\n\n"
                        "Decision:"
                    )
                    
                    response = requests.post(
                        self.groq_api_url,
                        headers=headers,
                        json={
                            "model": self.groq_model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant for meeting transcript segmentation."},
                                {"role": "user", "content": prompt}
                            ]
                        }
                    )
                    
                    if response.status_code != 200:
                        raise RuntimeError(f"LLM boundary evaluation failed: {response.status_code} - {response.text}")
                    
                    decision = response.json()["choices"][0]["message"]["content"].strip().lower()
                    logging.info(f"Boundary between segments {idx} and {idx+1}: LLM decision = {decision}")
                    
                    if "merge" in decision:
                        merged = left_segment + right_segment
                        refined_segments.append(merged)
                        idx += 2  # Skip the next segment as it has been merged
                        merge_occurred = True
                        continue
                    else:
                        refined_segments.append(left_segment)
                else:
                    refined_segments.append(segments[idx])
                idx += 1
            # If no merges occurred in this iteration, stop refining.
            if not merge_occurred:
                logging.info("No further merges; breaking out of refinement loop.")
                break
            segments = refined_segments
            iteration += 1
        
        logging.info(f"Total segments after iterative refinement: {len(segments)}")
        return segments

    def annotate_segments_with_topics(self, segments: List[List[Dict[str, any]]]) -> List[Dict[str, any]]:
        annotated = []
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        for idx, segment in enumerate(segments):
            text_block = "\n".join(f"{entry['speaker']}: {entry['utterance']}" for entry in segment)
            prompt = (
                "Summarize the main topic or theme of the following meeting discussion segment:\n\n"
                f"{text_block}\n\nTopic:"
            )
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

    def plot_umap(self, transcript: List[Dict[str, any]], labels: List[int]):
        """
        Instead of TSNE, UMAP is used here for dimensionality reduction to better preserve data structure.
        """
        embeddings = self.embed_utterances(transcript)
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set2", s=60)
        plt.colorbar(label="Cluster Label")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title("UMAP Visualization of Transcript Embeddings")
        plt.show()

    def plot_semantic_drift(self, transcript: List[Dict[str, any]], labels: List[int]):
        """
        Visualize semantic drift between consecutive utterances over time,
        with vertical lines indicating segment boundaries.
        """
        embeddings = self.embed_utterances(transcript)

        # Compute 1 - cosine similarity between each pair of consecutive utterances
        similarity = cosine_similarity(embeddings)
        drift = [1 - similarity[i, i + 1] for i in range(len(embeddings) - 1)]

        # Get segment change positions (where cluster label changes)
        boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]

        # Plot drift values
        plt.figure(figsize=(14, 6))
        plt.plot(drift, label="Semantic Drift", color="darkorange")
        for boundary in boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6)

        plt.xlabel("Utterance Index")
        plt.ylabel("Drift Score (1 - Cosine Similarity)")
        plt.title("Semantic Drift Over Time with Segment Boundaries")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


from scipy.signal import find_peaks

def detect_significant_drifts(drift_scores: List[float], threshold: float = 0.55) -> List[int]:
    # Find peaks above a semantic drift threshold
    peaks, _ = find_peaks(drift_scores, height=threshold, distance=3)
    return peaks.tolist()

def label_drift_changes(transcript: List[Dict[str, any]], peaks: List[int], window: int = 3, model="mistral-saba-24b") -> Dict[int, str]:
    drift_labels = {}
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    for idx in peaks:
        before_context = "\n".join(f"{entry['speaker']}: {entry['utterance']}" for entry in transcript[max(0, idx-window):idx])
        after_context = "\n".join(f"{entry['speaker']}: {entry['utterance']}" for entry in transcript[idx:idx+window])

        prompt = (
            "You are a meeting assistant analyzing a transcript for topic shifts.\n"
            "Given the text before and after a point, summarize the shift in topic.\n"
            "Respond concisely in the form: 'Shift from X → Y'.\n\n"
            f"Before:\n{before_context}\n\nAfter:\n{after_context}\n\nSummary of Topic Shift:"
        )

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You analyze meeting transcripts for topic drift."},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code == 200:
            label = response.json()["choices"][0]["message"]["content"].strip()
            drift_labels[idx] = label
        else:
            drift_labels[idx] = "(LLM error)"

    return drift_labels

def plot_drift_with_labels(drift_scores, peaks, labels, segment_boundaries=None):
    plt.figure(figsize=(20, 6))
    plt.plot(drift_scores, label="Semantic Drift", color="orange")

    for peak in peaks:
        plt.axvline(x=peak, color="red", linestyle="--", alpha=0.5)
        if peak in labels:
            plt.text(peak + 1, drift_scores[peak] + 0.02, labels[peak], rotation=30, fontsize=8, color="red")

    if segment_boundaries:
        for boundary in segment_boundaries:
            plt.axvline(x=boundary, color="gray", linestyle="--", alpha=0.3)

    plt.xlabel("Utterance Index")
    plt.ylabel("Drift Score (1 - Cosine Similarity)")
    plt.title("Semantic Drift Over Time with Detected Topic Shifts")
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    transcript = retrieve_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")

    sample_transcript = [
        {"speaker": "Speaker_0", "utterance": "Hello everyone, let's begin by discussing our roadmap."},
        {"speaker": "Speaker_1", "utterance": "Sure, I have some updates on the product launch and market strategy."},
        {"speaker": "Speaker_0", "utterance": "Great, let's also review our timelines and dependencies."},
        {"speaker": "Speaker_1", "utterance": "I am concerned about the delays in the supply chain affecting our launch."},
        {"speaker": "Speaker_0", "utterance": "Understood, we'll need to address that in the next meeting."},
        {"speaker": "Speaker_1", "utterance": "Also, we should consider feedback from our initial customers."},
        {"speaker": "Speaker_0", "utterance": "Exactly, customer feedback is essential to refine our approach."},
        {"speaker": "Speaker_1", "utterance": "Let's schedule a follow-up with the customer relations team."}
    ]


    segmenter = TranscriptSegmenter()
    
    # Improved clustering using HDBSCAN
    # best_labels, cluster_info = segmenter.cluster_transcript(sample_transcript)
    
    # # Assign segments based on the HDBSCAN labels (ignoring noise points with label -1)
    # segments = segmenter.assign_segments(sample_transcript, best_labels)
    
    # # Iteratively refine segments with LLM-based boundary evaluation.
    # refined_segments = segmenter.refine_segments_with_llm(segments, max_iterations=3)
    
    # # Annotate each refined segment with a topic label.
    # annotated = segmenter.annotate_segments_with_topics(refined_segments)
    
    # print("HDBSCAN Clustering Info:", cluster_info)
    # print("\nFinal Annotated Segments:")
    # for idx, annotated_seg in enumerate(annotated):
    #     print(f"\nSegment {idx} - Topic: {annotated_seg['topic']}")
    #     for entry in annotated_seg["segment"]:
    #         print(f"  {entry['speaker']}: {entry['utterance']}")
    
    # # Plot using UMAP for a better visual representation.
    # # segmenter.plot_umap(transcript, best_labels)
    # segmenter.plot_semantic_drift(sample_transcript, best_labels)

    embeddings = segmenter.embed_utterances(transcript)
    drift_scores = [cosine(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    # Step 2: Detect drift peaks
    peaks = detect_significant_drifts(drift_scores)

    # Step 3: Label with LLM
    drift_labels = label_drift_changes(transcript, peaks)

    # Step 4: Plot with annotated drift points
    segment_boundaries = [i for i, label in enumerate(best_labels) if label == -1]  # optional
    plot_drift_with_labels(drift_scores, peaks, drift_labels, segment_boundaries)
