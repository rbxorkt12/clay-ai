"""Database models for agent memory persistence."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Float,
    JSON,
    Integer,
    ForeignKey,
    Enum as SQLEnum,
    UUID,
)
from sqlalchemy.orm import relationship
from uuid import uuid4
import json

from core.database import Base
from models.tasks import TaskPriority, TaskStatus, TaskType
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


class MemoryEntryDB(Base):
    """Database model for memory entries."""

    __tablename__ = "memory_entries"

    id = Column(String(36), primary_key=True)
    agent_id = Column(String(36), ForeignKey("agents.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    content = Column(JSON)
    context = Column(String(255))
    importance = Column(Float)
    memory_type = Column(String(20), default="short_term")

    agent = relationship("AgentDB", back_populates="memories")


class TaskDB(Base):
    """Database model for tasks."""

    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    type = Column(SQLEnum(TaskType), nullable=False)
    priority = Column(
        SQLEnum(TaskPriority), nullable=False, default=TaskPriority.MEDIUM
    )
    status = Column(
        SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING
    )
    config = Column(JSON, nullable=False)
    input = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    assigned_agent_id = Column(
        String(36), ForeignKey("agents.id"), nullable=True
    )
    parent_task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    assigned_agent = relationship("AgentDB", back_populates="tasks")
    parent_task = relationship("TaskDB", remote_side=[id], backref="subtasks")


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
            # Check if entry already exists
            existing = await session.execute(
                select(MemoryEntryDB).where(MemoryEntryDB.id == entry["id"])
            )
            if existing.scalar_one_or_none() is None:
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
