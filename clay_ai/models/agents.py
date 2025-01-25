"""Agent models and schemas for Clay AI."""

from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .memory import MemoryEntryDB
from core.database import Base
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship


class AgentMemory:
    def __init__(self, capacity: int = 100, retention_policy: str = "fifo"):
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []
        self.capacity = capacity
        self.retention_policy = retention_policy

    def add(
        self,
        content: Dict[str, Any],
        context: str = "general",
        importance: float = 0.5,
    ):
        """Add a new memory entry."""
        entry = {
            "id": str(uuid4()),
            "timestamp": datetime.utcnow(),
            "content": content,
            "context": context,
            "importance": importance,
            "memory_type": "short_term",
        }

        self.short_term.append(entry)

        # Apply retention policy if capacity exceeded
        if len(self.short_term) > self.capacity:
            if self.retention_policy == "fifo":
                self.short_term.pop(0)
            elif self.retention_policy == "importance":
                least_important = min(
                    self.short_term, key=lambda x: x["importance"]
                )
                self.short_term.remove(least_important)

    async def save_to_db(self, session: AsyncSession, agent_id: str):
        """Save memory entries to database."""
        for entry in self.short_term + self.long_term:
            db_entry = MemoryEntryDB(
                id=entry["id"],
                agent_id=agent_id,
                timestamp=entry["timestamp"],
                content=entry["content"],
                context=entry["context"],
                importance=entry["importance"],
                memory_type=entry["memory_type"],
            )
            session.add(db_entry)

        await session.commit()

    async def load_from_db(self, session: AsyncSession, agent_id: str):
        """Load memory entries from database."""
        query = select(MemoryEntryDB).where(MemoryEntryDB.agent_id == agent_id)
        result = await session.execute(query)
        entries = result.scalars().all()

        for entry in entries:
            memory_entry = {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "content": entry.content,
                "context": entry.context,
                "importance": entry.importance,
                "memory_type": entry.memory_type,
            }
            if entry.memory_type == "short_term":
                self.short_term.append(memory_entry)
            else:
                self.long_term.append(memory_entry)


class AgentDB(Base):
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True)
    name = Column(String(100))
    config = Column(JSON)
    status = Column(String(20))
    current_task = Column(String(36), nullable=True)
    error = Column(String(255), nullable=True)
    metrics = Column(JSON)

    memories = relationship("MemoryEntryDB", back_populates="agent")
    tasks = relationship("TaskDB", back_populates="assigned_agent")


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
        default_factory=dict, description="Capability parameters"
    )


class AgentConfig(BaseModel):
    """Agent configuration settings."""

    role: AgentRole = Field(..., description="Agent role")
    capabilities: List[AgentCapability] = Field(
        default_factory=list, description="Agent capabilities"
    )
    memory_limit: int = Field(default=1024, description="Memory limit in MB")
    priority: int = Field(default=1, description="Agent priority (1-10)")
    timeout: int = Field(default=300, description="Task timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum task retries")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration parameters"
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
        default=AgentStatus.INITIALIZING, description="Current agent status"
    )
    current_task: Optional[str] = Field(
        default=None, description="Current task ID"
    )
    metrics: AgentMetrics = Field(
        default_factory=AgentMetrics, description="Agent metrics"
    )
    memory: AgentMemory = Field(
        default_factory=AgentMemory, description="Agent memory storage"
    )
    error: Optional[str] = Field(
        default=None, description="Last error message"
    )

    model_config = {"arbitrary_types_allowed": True}


class AgentCreate(BaseModel):
    """Schema for creating an agent."""

    name: str = Field(..., description="Agent name")
    config: AgentConfig = Field(..., description="Agent configuration")


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""

    name: Optional[str] = Field(None, description="Agent name")
    config: Optional[AgentConfig] = Field(
        None, description="Agent configuration"
    )


class AgentResponse(BaseModel):
    """Schema for agent responses."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    agent: Optional[Agent] = Field(None, description="Agent data")
