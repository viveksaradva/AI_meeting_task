from groq import Groq  
from dotenv import load_dotenv
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_groq_response(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq Error: {e}")
        return "{}"  # Safe fallback
