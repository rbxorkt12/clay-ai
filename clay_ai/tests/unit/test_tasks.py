"""Unit tests for task management API."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from clay_ai.api.tasks import (
    create_task,
    list_tasks,
    get_task,
    update_task,
    delete_task
)
from clay_ai.models.tasks import (
    Task,
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskType,
    TaskStatus,
    TaskPriority
)


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def sample_task_data():
    """Create sample task data for testing."""
    return {
        "id": "test-task-1",
        "type": TaskType.ANALYSIS,
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.PENDING,
        "input_data": {
            "code": "print('hello world')"
        }
    }


@pytest.mark.asyncio
async def test_create_task(mock_db, sample_task_data):
    """Test task creation endpoint."""
    task_data = TaskCreate(
        type=sample_task_data["type"],
        input_data=sample_task_data["input_data"]
    )
    
    response = await create_task(task_data, mock_db)
    assert isinstance(response, TaskResponse)
    assert response.success is True
    assert "created successfully" in response.message.lower()


@pytest.mark.asyncio
async def test_list_tasks(mock_db):
    """Test task listing endpoint."""
    tasks = await list_tasks(None, mock_db)
    assert isinstance(tasks, list)


@pytest.mark.asyncio
async def test_list_tasks_with_status(mock_db):
    """Test task listing with status filter."""
    tasks = await list_tasks(TaskStatus.PENDING, mock_db)
    assert isinstance(tasks, list)


@pytest.mark.asyncio
async def test_get_task(mock_db, sample_task_data):
    """Test task retrieval endpoint."""
    response = await get_task("test-task-1", mock_db)
    assert isinstance(response, TaskResponse)
    assert response.success is True
    assert "retrieved successfully" in response.message.lower()


@pytest.mark.asyncio
async def test_update_task(mock_db, sample_task_data):
    """Test task update endpoint."""
    task_data = TaskUpdate(status=TaskStatus.RUNNING)
    response = await update_task("test-task-1", task_data, mock_db)
    assert isinstance(response, TaskResponse)
    assert response.success is True
    assert "updated successfully" in response.message.lower()


@pytest.mark.asyncio
async def test_delete_task(mock_db):
    """Test task deletion endpoint."""
    response = await delete_task("test-task-1", mock_db)
    assert isinstance(response, TaskResponse)
    assert response.success is True
    assert "deleted successfully" in response.message.lower()


@pytest.mark.asyncio
async def test_task_not_found(mock_db):
    """Test error handling for non-existent task."""
    mock_db.get.side_effect = Exception("Task not found")
    
    with pytest.raises(HTTPException) as exc_info:
        await get_task("non-existent-task", mock_db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_invalid_task_data(mock_db):
    """Test error handling for invalid task data."""
    invalid_task = TaskCreate(type=TaskType.ANALYSIS)  # Missing required data
    
    with pytest.raises(HTTPException) as exc_info:
        await create_task(invalid_task, mock_db)
    assert exc_info.value.status_code == 500 