"""MM-RAG Backend — FastAPI entry point.

Run with:
    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI

from backend.config import get_settings

settings = get_settings()

app = FastAPI(
    title="MM-RAG API",
    description="Multi-Modal Multi-Document Retrieval-Augmented Generation",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health/status endpoint to verify the backend is running."""
    return {
        "status": "healthy",
        "environment": settings.environment,
    }
