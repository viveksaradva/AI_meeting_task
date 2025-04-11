import time
import json
import logging
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
from modules.db.postgres import retrieve_transcript

# Import your Groq API utility function; it takes (prompt, model) and returns a response string.
from modules.llm import get_groq_response

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class HighLevelMeetingAnalyzer:
    def __init__(self, transcript: List[Dict[str, any]]):
        """
        transcript: List of dicts with keys: 'speaker', 'start', 'end', 'utterance'
        """
        self.transcript = transcript
        # Embedding model for topic extraction and drift detection
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Sentiment analysis pipeline (using HuggingFace as fallback); adjust as needed
        self.sentiment_pipeline = hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        # Define Groq models: primary for analysis and a fallback alternative
        self.primary_model = "llama-3.3-70b-versatile"   # replace with the actual Groq model ID (e.g., for summarization)
        self.fallback_model = "gpt-4"  # fallback model if primary is not appropriate

    def get_intro_segment(self, duration_sec: float = 120.0) -> List[Dict[str, any]]:
        """Return transcript entries from the first 'duration_sec' seconds."""
        return [entry for entry in self.transcript if entry['start'] <= duration_sec]

    def segment_transcript(self, segment_duration: float = 120.0) -> List[List[Dict[str, any]]]:
        """Splits transcript into segments based on time duration."""
        segments = []
        current_segment = []
        current_start = self.transcript[0]['start']
        for entry in self.transcript:
            if entry['end'] - current_start <= segment_duration:
                current_segment.append(entry)
            else:
                segments.append(current_segment)
                current_segment = [entry]
                current_start = entry['start']
        if current_segment:
            segments.append(current_segment)
        return segments

    def extract_agenda(self, intro_segment: List[Dict[str, any]]) -> List[str]:
        """
        Uses Groq API to extract agenda items from the introduction segment.
        Returns a list of agenda items.
        """
        text = " ".join(entry['utterance'] for entry in intro_segment)
        prompt = (
            f"Extract the clear and distinct agenda items from the following meeting introduction. "
            f"Return each agenda item on a new line.\n\nText:\n{text}\n\nAgenda items:"
        )
        try:
            response = get_groq_response(prompt, self.primary_model)
        except Exception as e:
            logging.error(f"Primary Groq model failed: {e}. Falling back.")
            response = get_groq_response(prompt, self.fallback_model)
        agenda_items = [line.strip() for line in response.splitlines() if line.strip()]
        return agenda_items

    def summarize_segment_topic(self, segment: List[Dict[str, any]]) -> str:
        """
        Uses Groq API to summarize the main topic of a transcript segment.
        Returns a concise sentence summarizing the segment's topic.
        """
        text = " ".join(entry['utterance'] for entry in segment)
        prompt = (
            f"Summarize the main topic of the following conversation segment in one clear sentence:\n\n{text}\n\nSummary:"
        )
        try:
            summary = get_groq_response(prompt, self.primary_model)
        except Exception as e:
            logging.error(f"Primary Groq model failed: {e}. Falling back.")
            summary = get_groq_response(prompt, self.fallback_model)
        return summary.strip()

    def detect_topic_drift(self, agenda: List[str], segment_topics: List[str]) -> Dict[str, float]:
        """
        Compares each segment's topic to the agenda and returns a drift score per segment.
        Drift score: 0.0 indicates full alignment, 1.0 indicates complete divergence.
        """
        agenda_text = " ".join(agenda)
        agenda_embedding = self.embed_model.encode(agenda_text)
        drift_scores = {}
        for idx, topic in enumerate(segment_topics):
            topic_embedding = self.embed_model.encode(topic)
            sim = util.cos_sim(agenda_embedding, topic_embedding)[0][0].item()
            drift_scores[f"segment_{idx}"] = round(1 - sim, 2)
        time.sleep(18)
        return drift_scores

    def infer_participant_goals(self) -> Dict[str, str]:
        """
        Uses Groq API to infer if each participant achieved their objectives.
        Returns a JSON mapping: e.g., { "speaker_0": "achieved", "speaker_1": "not achieved" }.
        """
        text = "\n".join([f"{t['speaker']}: {t['utterance']}" for t in self.transcript])
        prompt = (
            "Based solely on the following transcript, determine for each speaker whether they achieved their objectives by the end of the meeting. "
            "Return your answer in strict JSON format, mapping each speaker to either \"achieved\" or \"not achieved\".\n\n"
            f"Transcript:\n{text}\n\n"
            "Example Output:\n{\n  \"speaker_0\": \"achieved\",\n  \"speaker_1\": \"not achieved\"\n}"
        )
        try:
            response = get_groq_response(prompt, self.primary_model)
        except Exception as e:
            logging.error(f"Primary Groq model failed: {e}. Falling back.")
            response = get_groq_response(prompt, self.fallback_model)
        try:
            return json.loads(response.strip())
        except Exception as e:
            logging.error(f"Failed to parse participant goals: {e}")
            return {}
        
    def analyze_sentiment(self) -> Dict[str, float]:
        """
        Analyzes overall sentiment per speaker using the sentiment analysis pipeline.
        Returns a dictionary mapping each speaker to their average sentiment score.
        """
        sentiment_map = {}
        count_map = {}
        for entry in self.transcript:
            speaker = entry['speaker']
            score = self.sentiment_pipeline(entry['utterance'])[0]['score']
            sentiment_map[speaker] = sentiment_map.get(speaker, 0) + score
            count_map[speaker] = count_map.get(speaker, 0) + 1
        return {speaker: sentiment_map[speaker] / count_map[speaker] for speaker in sentiment_map}
    
    def run_analysis(self) -> Dict:
        logging.info("Starting high-level meeting analysis...")
        
        # Extract agenda items from the introduction (first 2 minutes)
        intro_segment = self.get_intro_segment(duration_sec=120)
        agenda_items = self.extract_agenda(intro_segment)
        logging.info(f"Extracted Agenda Items: {agenda_items}")
        
        # Segment transcript and get topics per segment
        segments = self.segment_transcript(segment_duration=120)
        logging.info(f"Total segments: {len(segments)}")
        segment_topics = []
        for seg in tqdm(segments, desc="Summarizing segments", unit="segment"):
            topic = self.summarize_segment_topic(seg)
            segment_topics.append(topic) 

        # Detect topic drift compared to the agenda
        drift_scores = self.detect_topic_drift(agenda_items, segment_topics)
        logging.info(f"Drift Scores: {drift_scores}")
        
        # Infer participant goals for the full transcript
        goals = self.infer_participant_goals()
        logging.info(f"Participant Goals: {goals}")
        
        # Analyze overall sentiment per speaker
        sentiment_summary = self.analyze_sentiment()
        logging.info(f"Sentiment Summary: {sentiment_summary}")
        
        analysis = {
            "agenda_items": agenda_items,
            "segment_topics": segment_topics,
            "topic_drift": drift_scores,
            "participant_goals": goals,
            "sentiment_summary": sentiment_summary,
        }
        return analysis

# Example usage:
if __name__ == "__main__":
    # Assuming extended_transcript is provided in modules/db/sample_data.py
    transcript = retrieve_transcript(meeting_id="1f253dcb-2838-48b9-8ef1-73e70259f116")
    analyzer = HighLevelMeetingAnalyzer(transcript)
    analysis_result = analyzer.run_analysis()
    print(json.dumps(analysis_result, indent=2))
