from modules.pipelines.speaker_diarization_based_transcription_pipeline import SpeechProcessingPipeline
from modules.db.postgres import insert_transcript
from modules.prompts import identify_speaker_role_prompt, format_transcript_for_roles
from modules.llm import get_groq_response
import json

class SpeakerRoleInferencePipeline:
    def __init__(self, audio_file_path: str):
        self.audio_file_path = audio_file_path

    def run(self):
        transcript = self.diarize_and_transcribe(self.audio_file_path)
        samples = self.sample_utterances(transcript)
        role_mapping = self.identify_roles(samples)
        enriched_transcript = self.label_full_transcript(transcript, role_mapping)
        self.insert_to_db(enriched_transcript)
        return enriched_transcript

    def diarize_and_transcribe(self, audio_path):
        return SpeechProcessingPipeline(audio_path).run_pipeline()

    def sample_utterances(self, transcript, max_per_speaker=3):
        """
        Samples up to `max_per_speaker` utterances per speaker and returns a flat list
        of dicts with speaker and utterance keys.
        """
        utterance_count = {}
        samples = []

        for entry in transcript:
            speaker = entry["speaker"]
            if speaker not in utterance_count:
                utterance_count[speaker] = 0

            if utterance_count[speaker] < max_per_speaker:
                samples.append({
                    "speaker": speaker,
                    "utterance": entry["utterance"]
                })
                utterance_count[speaker] += 1

        return samples

    def identify_roles(self, samples):
        # return infer_speaker_roles(samples)  # Use your Groq LLM logic
        formatted = format_transcript_for_roles(samples)
        prompt = identify_speaker_role_prompt(formatted)
        
        # logger.info("Calling Groq LLM to classify speaker roles...")
        raw_response = get_groq_response(prompt)

        try:
            role_mapping = json.loads(raw_response)
            return role_mapping
        except json.JSONDecodeError:
            # logger.error("LLM returned malformed JSON. Response:\n" + raw_response)
            raise

    def label_full_transcript(self, transcript, role_mapping):
        return [
            {**entry, "speaker": role_mapping.get(f"Speaker_{entry['speaker'].split('_')[1]}", entry['speaker'])}
            for entry in transcript
        ]

    def insert_to_db(self, enriched_transcript):
        insert_transcript(enriched_transcript)