"""Task models and schemas for Clay AI."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID, uuid4


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status states."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Available task types."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    OBSERVATION = "observation"
    LINT = "lint"
    CRAWL = "crawl"
    STORE = "store"


class TaskResult(BaseModel):
    """Task execution result."""

    success: bool = Field(default=False)
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "output": {"total_files": 2, "fixed_errors": 5},
                "metrics": {"execution_time": 1.23, "memory_used": 256},
            }
        }
    )


class TaskConfig(BaseModel):
    """Task configuration settings."""

    timeout: int = 300
    max_retries: int = 3
    required_capabilities: List[str] = Field(default_factory=list)
    memory_requirement: int = 512


class Task(BaseModel):
    """Core task model."""

    id: UUID = Field(default_factory=uuid4)
    type: TaskType = Field(default=TaskType.ANALYSIS)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    config: TaskConfig = Field(default_factory=TaskConfig)
    input: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[TaskResult] = None
    assigned_agent: Optional[str] = None
    parent_task: Optional[UUID] = None
    subtasks: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "type": "lint",
                "input": {
                    "files": ["file1.py", "file2.py"],
                    "error_types": ["E501", "F401"],
                },
                "assigned_agent": "linter-1",
                "started_at": "2024-01-24T12:00:00Z",
            }
        }
    )


class TaskCreate(BaseModel):
    """Schema for creating a task."""

    type: TaskType = Field(..., description="Task type")
    priority: Optional[TaskPriority] = None
    config: Optional[TaskConfig] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    parent_task: Optional[str] = None


class TaskUpdate(BaseModel):
    """Schema for updating a task."""

    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    config: Optional[TaskConfig] = None
    input_data: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None


class TaskResponse(BaseModel):
    """Schema for task responses."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    task: Optional[Task] = Field(None, description="Task data")


Task.model_rebuild()  # Update forward refs
