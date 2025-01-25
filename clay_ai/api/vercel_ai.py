"""
API endpoints for Vercel AI integration.
Provides streaming code execution capabilities using Vercel AI and E2B sandbox.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from vercel_ai_core import Message

from clay_ai.core.vercel_ai import vercel_ai_bridge
from clay_ai.core.config import settings

router = APIRouter(prefix="/api/v1/vercel-ai")


@router.post("/execute")
async def execute_code(
    messages: List[Message],
    model_params: Dict | None = None,
) -> StreamingTextResponse:
    """
    Execute code using Vercel AI streaming and E2B sandbox.
    
    Args:
        messages: List of chat messages containing code to execute
        model_params: Optional parameters for model configuration
        
    Returns:
        Streaming response with execution results
    """
    try:
        return await vercel_ai_bridge.execute_with_streaming(
            messages=messages,
            model_params=model_params
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing code: {str(e)}"
        ) 