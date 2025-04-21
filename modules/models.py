from pydantic import BaseModel
from typing import List, Optional

class Utterance(BaseModel):
    speaker: str
    start: float
    end: float
    utterance: str

class Segment(BaseModel):
    segment_id: int
    start: float             # segment start time in seconds
    end: float               # segment end time in seconds
    speakers: List[str]      # unique list of speakers in this segment
    summary: Optional[str] = None
    confidence: Optional[float]  # None for segment_0, otherwise boundary confidence
    utterances: List[Utterance]

class DriftPoint(BaseModel):
    start: float             # time (s) where new topic begins
    confidence: float        # same as the segment’s boundary_confidence
    prev_snippet: str        # last utterance of the prior segment, prefixed by speaker
    next_snippet: str        # first utterance of the new segment, prefixed by speaker
    segment_before: int      # index of the segment before the drift
    segment_after: int       # index of the segment after the drift

class SegmentationResponse(BaseModel):
    segments: List[Segment]
    drift_points: List[DriftPoint]
