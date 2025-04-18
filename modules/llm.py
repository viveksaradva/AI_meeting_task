from groq import Groq
from dotenv import load_dotenv
import os
import logging
from ratelimit import limits, sleep_and_retry

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

logger = logging.getLogger(__name__)

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute (60 seconds)
def get_groq_response(prompt: str, model: str) -> str:
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    return response.choices[0].message.content
