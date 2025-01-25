"""Response models using PydanticAI for structured validation."""

from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, AgentResponse

T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model with success and message."""
    
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    message: str
    error: Optional[str] = None


class DataResponse(BaseResponse, Generic[T]):
    """Generic response model with data payload."""
    
    data: Optional[T] = None


class AgentTaskResponse(AgentResponse):
    """Agent task response with execution metrics."""
    
    execution_time: float
    memory_usage: float
    output: Dict[str, Any]
    error: Optional[str] = None


class StreamResponse(BaseResponse):
    """Streaming response model for real-time updates."""
    
    event: str
    data: Dict[str, Any]
    sequence: int 