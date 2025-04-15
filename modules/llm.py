from groq import Groq
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception, before_sleep_log
import logging

logger = logging.getLogger(__name__)

def is_rate_limited(exception):
    return hasattr(exception, "response") and getattr(exception.response, "status_code", None) == 429

@retry(
    retry=retry_if_exception(is_rate_limited),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
@sleep_and_retry
@limits(calls=5, period=10)
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
