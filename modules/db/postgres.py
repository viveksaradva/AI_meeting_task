import os
import uuid
import logging
import psycopg2
from contextlib import contextmanager
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
        )
        yield conn
        conn.commit()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

def insert_transcript(transcript: List[Dict], meeting_id: str = None):
    """
    Inserts transcript into the `meeting_transcripts` table.

    Each entry must contain: speaker, role, start, end, transcription.
    """
    meeting_id = meeting_id or str(uuid.uuid4())

    logger.info(f"Inserting transcript for meeting_id: {meeting_id}")

    query = """
    INSERT INTO meeting_transcripts (meeting_id, speaker_label, utterance, start, "end")
    VALUES (%s, %s, %s, %s, %s)
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        for entry in transcript:
            try:
                cursor.execute(query, (
                    meeting_id,
                    entry.get("speaker"),
                    entry.get("utterance"),
                    entry.get("start"),
                    entry.get("end")
                ))
            except Exception as e:
                logger.error(f"Insert failed for entry {entry}: {e}")
        cursor.close()

    logger.info(f"Inserted {len(transcript)} rows for meeting {meeting_id}")
    return meeting_id

def retrieve_transcript(meeting_id: str) -> List[Dict]:
    """
    Retrieves transcript from the `meeting_transcripts` table for a given meeting_id.

    Returns a list of dictionaries with keys: 'speaker', 'start', 'end', 'utterance'.
    """
    query = """
    SELECT speaker_label, start, "end", utterance
    FROM meeting_transcripts
    WHERE meeting_id = %s
    """

    transcript = []
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, (meeting_id,))
            rows = cursor.fetchall()
            for row in rows:
                transcript.append({
                    'speaker': row[0],
                    'start': row[1],
                    'end': row[2],
                    'utterance': row[3]
                })
        except Exception as e:
            logger.error(f"Retrieval failed for meeting_id {meeting_id}: {e}")
        cursor.close()

    logger.info(f"Retrieved {len(transcript)} rows for meeting {meeting_id}")
    return transcript

if __name__ == "__main__":
    print(retrieve_transcript(meeting_id="8753b367-ae49-4763-a860-64663b34ef83"))