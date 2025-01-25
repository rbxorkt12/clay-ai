"""
Main application module for Clay AI.
Integrates FastAPI with Clay AI components and Vercel AI capabilities.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from clay_ai.api import agents, tasks, events, vercel_ai
from clay_ai.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="Clay AI",
    description="Multi-Agent Framework for Intelligent Automation",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(agents.router)
app.include_router(tasks.router)
app.include_router(events.router)
app.include_router(vercel_ai.router)

# Health check endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"} 