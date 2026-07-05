from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from backend.agents.research_planner import plan_research_queries
from backend.agents.research_agent import run_research
import json, asyncio

router = APIRouter()

class ResearchRequest(BaseModel):
    topic:  str
    max_queries: int = 4

class ResearchResponse(BaseModel):
    topic:   str
    report:  str
    sources: list
    queries: list
    duration: float

@router.post("/",  response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """Standard research endpoint — returns complete report."""
    print(f"[Research] Starting research on: {req.topic}")

    # step 1: plan sub-queries
    queries = plan_research_queries(req.topic)
    queries = queries[:req.max_queries]

    # step 2: run full research pipeline
    result = run_research(req.topic, queries)

    return ResearchResponse(
        topic=req.topic,
        report=result["report"],
        sources=result["sources"],
        queries=queries,
        duration=result["duration"]
    )

@router.post("/stream")
async def research_stream(req: ResearchRequest):
    """
    SSE streaming research endpoint.
    Streams progress events as research happens.
    Frontend shows live status updates.
    """
    async def event_generator():
        # event 1: started
        yield f"data: {json.dumps({'event': 'started', 'topic': req.topic})}\n\n"
        await asyncio.sleep(0)

        # event 2: planning queries
        yield f"data: {json.dumps({'event': 'planning', 'message': 'Breaking topic into sub-queries...'})}\n\n"
        await asyncio.sleep(0)

        queries = plan_research_queries(req.topic)
        queries = queries[:req.max_queries]

        yield f"data: {json.dumps({'event': 'queries_ready', 'queries': queries})}\n\n"
        await asyncio.sleep(0)

        # event 3: searching
        for i, query in enumerate(queries):
            yield f"data: {json.dumps({'event': 'searching', 'query': query, 'index': i+1, 'total': len(queries)})}\n\n"
            await asyncio.sleep(0)

        # event 4: run full pipeline
        yield f"data: {json.dumps({'event': 'researching', 'message': 'Analyzing sources and generating report...'})}\n\n"
        await asyncio.sleep(0)

        result = run_research(req.topic, queries)

        # event 5: done
        yield f"data: {json.dumps({'event': 'complete', 'report': result['report'], 'sources': result['sources'], 'queries': queries, 'duration': result['duration']})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
