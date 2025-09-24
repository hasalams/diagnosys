from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from src.pipeline import ContentIntelligencePipeline

app = FastAPI()
pipeline = ContentIntelligencePipeline()

class RequestPayload(BaseModel):
    request: str

@app.post("/process_request")
async def process_request(payload: RequestPayload) -> Dict[str, Any]:
    """
    Process a content intelligence request via the pipeline.
    """
    result = await pipeline.process_request(payload.request)
    return result

# For local testing: uvicorn backend:app --reload
