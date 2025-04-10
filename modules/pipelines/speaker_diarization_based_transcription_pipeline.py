import os
import glob
import json
import shutil
import logging
import subprocess
import tempfile
import torchaudio
import urllib.request
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from omegaconf import OmegaConf
from transformers import pipeline as hf_pipeline
from nemo.collections.asr.models import ClusteringDiarizer
from modules.utils.output_suppression_utils import suppress_output, silence_transformers

load_dotenv()

# Configure logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class SpeechProcessingPipeline:
    """
    Modular speech processing pipeline:
    1. Converts audio to WAV
    2. Runs speaker diarization (NeMo)
    3. Transcribes each segment (Whisper)
    4. Returns diarized transcript
    """

    def __init__(self, input_audio: str, num_speakers: int = 2, model: str = "medium"):
        self.input_audio = Path(input_audio)
        self.num_speakers = num_speakers
        self.model = model
        self.audio_stem = self.input_audio.stem
        self.wav_file = None
        self.rttm_file = None
        self.diarized_transcript = None

    def run_pipeline(self) -> List[Dict[str, str]]:
        self._convert_to_wav()
        self._run_diarization()
        self._locate_rttm()
        transcript = self._transcribe_segments()
        self._cleanup()
        return transcript
    
    def _convert_to_wav(self):
        """Converts to 16kHz mono WAV if not already."""
        if self.input_audio.suffix == ".wav":
            self.wav_file = str(self.input_audio)
            logging.info("Input is already in WAV format.")
            return

        self.wav_file = f"{self.audio_stem}.wav"
        logging.info("Converting to WAV...")
        command = f"ffmpeg -i {self.input_audio} -ar 16000 -ac 1 {self.wav_file} -y"
        subprocess.run(command, shell=True, check=True)

    def _run_diarization(self):
        """Runs NeMo Clustering Diarizer."""
        config_path = self._ensure_diarization_config()
        manifest = {
            "audio_filepath": self.wav_file,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": self.num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None
        }

        with open("manifest.json", "w") as f:
            json.dump(manifest, f)
            f.write("\n")

        config = OmegaConf.load(config_path)
        config.diarizer.manifest_filepath = "manifest.json"
        config.diarizer.out_dir = "./"
        config.diarizer.speaker_embeddings.model_path = "titanet_large"

        logging.info("Running diarization...")
        with suppress_output():
            diarizer = ClusteringDiarizer(cfg=config)
            diarizer.diarize()

    def _locate_rttm(self):
        """Finds the generated RTTM file."""
        matches = glob.glob(f"**/{self.audio_stem}.rttm", recursive=True)
        if not matches:
            raise FileNotFoundError("RTTM not found.")
        self.rttm_file = matches[0]
        logging.info(f"Found RTTM: {self.rttm_file}")

    def _transcribe_segments(self) -> List[Dict[str, str]]:
        """Transcribes segments from RTTM."""
        waveform, sr = torchaudio.load(self.wav_file)
        segments = self._parse_rttm()

        silence_transformers()
        asr = hf_pipeline("automatic-speech-recognition", model=f"openai/whisper-{self.model}")
        results = []

        logging.info("Transcribing segments...")
        for seg in tqdm(segments, desc="Transcribing", unit="segment"):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            audio_chunk = waveform[:, start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                torchaudio.save(tmp.name, audio_chunk, sr)
                with suppress_output():
                    transcription = asr(tmp.name).get("text", "").strip()

            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "utterance": transcription
            })

        self.diarized_transcript = results
        return results

    def _parse_rttm(self) -> List[Dict[str, float]]:
        """Extracts segment info from RTTM."""
        segments = []
        with open(self.rttm_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                segments.append({
                    "speaker": parts[7],
                    "start": start,
                    "end": start + duration
                })
        return segments
    
    def _cleanup(self):
        files_to_delete = [
            "manifest.json",
            "manifest_vad_input.json",
            self.wav_file,
            self.rttm_file
        ]
        folders_to_delete = [
            "vad_outputs",
            "speaker_outputs",
            "pred_rttms"
        ]

        for file in files_to_delete:
            if file and os.path.exists(file):
                os.remove(file)

        for folder in folders_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        logging.info("Temporary files and folders cleaned up.")

    @staticmethod
    def _ensure_diarization_config() -> str:
        """Downloads NeMo config if missing."""
        path = "diar_infer_telephonic.yaml"
        url = os.getenv("DIARIZATION_CONFIG_URL")
        if not os.path.exists(path):
            logging.info("Downloading diarization config...")
            urllib.request.urlretrieve(url, path)
        return path

if __name__ == "__main__":
    pipeline = SpeechProcessingPipeline(input_audio="batman.mp3", num_speakers=2, model="base")
    result = pipeline.run_pipeline()
    print(result)