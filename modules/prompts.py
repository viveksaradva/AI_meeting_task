from typing import List, Dict

def identify_speaker_role_prompt(formatted_transcript: str) -> str:
    return f"""
    You are a precise and analytical meeting role classifier. Your task is to determine the most likely real-world role for each speaker solely based on their dialogue in the transcript below.

    Consider roles such as "Product Manager", "Client Lead", "Sales Executive", "Technical Engineer", "CTO", etc. Analyze the language and context of the conversation to infer the roles.

    **IMPORTANT:** Return ONLY the JSON mapping in the exact format below. DO NOT include any additional text, explanations, or commentary.

    **Return a single role for each speaker.** If multiple roles seem applicable, choose the most appropriate one, but avoid using slashes ("/") or multiple roles.

    Expected JSON format:
    {{
    "Speaker_0": "<role>",
    "Speaker_1": "<role>"
    }}

    Transcript:
    {formatted_transcript}
    """.strip()

def format_transcript_for_roles(transcript: List[Dict[str, str]]) -> str:
    return "\n".join([f"{seg['speaker']}: {seg['utterance']}" for seg in transcript])
