import logging
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from typing import Dict, List
from modules.models import Utterance
from modules.models import SegmentationResponse, Segment, Utterance, DriftPoint
from modules.pipelines.topic_drift_analyzer_algo import SegmenterAlgorithm

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Simple summarizer—for production you might swap in your own LLM chain
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TranscriptInput(BaseModel):
    text: List[Utterance]   # The large transcript text submitted by the user

# @app.get("/segment/{meeting_id}", response_model=SegmentationResponse)
# def segment_meeting(meeting_id: str):
#     # 1. Load & run segmentation
#     seg_algo = SegmenterAlgorithm()
#     seg_algo.load_transcript(meeting_id)
@app.post("/segment", response_model=SegmentationResponse)
def segment_meeting(input_data: TranscriptInput):
    """
    Accept a raw transcript text, process segmentation, and identify topic drift.
    """
    seg_algo = SegmenterAlgorithm()

    # 1. Set the transcript from the user input
    seg_algo.transcript = [u.dict() for u in input_data.text]
    try:
        seg_algo.embed_transcript()
        seg_algo.cluster_transcript()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/Clustering failed: {e}")

    # 2. Build & validate segments
    raw_segs = seg_algo.build_segments_from_labels()
    validated = seg_algo.validate_segments_with_llm(raw_segs)

    response_segments: List[Segment] = []
    drift_points: List[DriftPoint] = []

    # 3. Turn algorithm output into our API schema
    for idx, seg in enumerate(validated):
        utts = seg["utterances"]
        # compute segment start/end
        starts = [u["start"] for u in utts]
        ends   = [u["end"]   for u in utts]
        seg_start, seg_end = min(starts), max(ends)
        # collect speakers
        speakers = list({u["speaker"] for u in utts})
        # generate a summary
        text_blob = " ".join(u["utterance"] for u in utts)
        # summary = summarizer(text_blob, max_length=30, min_length=5)[0]["summary_text"]
        # boundary confidence (None for first)
        confidence = seg.get("boundary_confidence", None)

        # build Pydantic Segment
        response_segments.append(
            Segment(
                segment_id=idx,
                start=seg_start,
                end=seg_end,
                speakers=speakers,
                # summary=summary,
                confidence=confidence,
                utterances=[Utterance(**u) for u in utts]
            )
        )

        # if this is not the first segment, add a drift_point
        if idx > 0 and confidence is not None:
            prev_utt = response_segments[idx-1].utterances[-1]
            curr_utt = response_segments[idx].utterances[0]
            drift_points.append(
                DriftPoint(
                    start=seg_start,
                    confidence=confidence,
                    prev_snippet=f"{prev_utt.speaker}: {prev_utt.utterance}",
                    next_snippet=f"{curr_utt.speaker}: {curr_utt.utterance}",
                    segment_before=idx-1,
                    segment_after=idx
                )
            )

    return SegmentationResponse(segments=response_segments, drift_points=drift_points)
