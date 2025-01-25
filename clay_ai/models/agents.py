"""Agent models and schemas for Clay AI."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Available agent roles."""
    EXECUTOR = "executor"
    PLANNER = "planner"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"
    CRITIC = "critic"


class AgentStatus(str, Enum):
    """Possible agent states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(BaseModel):
    """Agent capability definition."""
    name: str = Field(..., description="Capability name")
    version: str = Field(..., description="Capability version")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Capability parameters"
    )


class AgentConfig(BaseModel):
    """Agent configuration settings."""
    role: AgentRole = Field(..., description="Agent role")
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Agent capabilities"
    )
    memory_limit: int = Field(
        default=1024,
        description="Memory limit in MB"
    )
    priority: int = Field(
        default=1,
        description="Agent priority (1-10)"
    )
    timeout: int = Field(
        default=300,
        description="Task timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum task retries"
    )


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    tasks_completed: int = Field(default=0)
    tasks_failed: int = Field(default=0)
    avg_response_time: float = Field(default=0.0)
    memory_usage: int = Field(default=0)
    cpu_usage: float = Field(default=0.0)
    uptime: int = Field(default=0)


class Agent(BaseModel):
    """Core agent model."""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    config: AgentConfig = Field(..., description="Agent configuration")
    status: AgentStatus = Field(
        default=AgentStatus.INITIALIZING,
        description="Current agent status"
    )
    current_task: Optional[str] = Field(
        default=None,
        description="Current task ID"
    )
    metrics: AgentMetrics = Field(
        default_factory=AgentMetrics,
        description="Agent metrics"
    )
    error: Optional[str] = Field(
        default=None,
        description="Last error message"
    )


class AgentCreate(BaseModel):
    """Schema for creating an agent."""
    name: str = Field(..., description="Agent name")
    config: AgentConfig = Field(..., description="Agent configuration")


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""
    name: Optional[str] = Field(None, description="Agent name")
    config: Optional[AgentConfig] = Field(
        None,
        description="Agent configuration"
    )


class AgentResponse(BaseModel):
    """Schema for agent responses."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    agent: Optional[Agent] = Field(None, description="Agent data") 