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
    """
    Context manager to handle a Postgres database connection.

    The connection is established using the environment variables
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST and POSTGRES_PORT.

    If the connection fails, an exception is raised.

    The connection is committed after the context is exited normally.

    The connection is closed at the end of the context, whether it was exited normally or not.
    """
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
    Inserts transcript into the `test_meeting_transcripts` table.

    Each entry must contain: speaker, utterance, start, end.
    """
    meeting_id = meeting_id or str(uuid.uuid4())

    logger.info(f"Inserting test transcript for meeting_id: {meeting_id}")

    query = """
    INSERT INTO test_meeting_transcripts (meeting_id, speaker, utterance, start_time, end_time)
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

    logger.info(f"Inserted {len(transcript)} rows into test_meeting_transcripts for meeting {meeting_id}")
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

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT speaker, utterance, start_time, end_time
            FROM test_meeting_transcripts
            WHERE meeting_id = %s
            ORDER BY start_time ASC
        """, (meeting_id,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            {
                "speaker": row[0],
                "utterance": row[1],
                "start": float(row[2]),
                "end": float(row[3])
            }
            for row in rows
        ]