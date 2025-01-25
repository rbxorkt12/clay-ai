"""Unit tests for executor agent."""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from clay_ai.agents.executor import ExecutorAgent
from clay_ai.models.agents import AgentConfig, AgentRole, AgentStatus
from clay_ai.models.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskPriority,
    TaskConfig
)


@pytest.fixture
def agent_config():
    """Create sample agent configuration."""
    return AgentConfig(
        role=AgentRole.EXECUTOR,
        capabilities=[],
        memory_limit=1024,
        timeout=300
    )


@pytest.fixture
def mock_sandbox():
    """Create mock sandbox instance."""
    sandbox = AsyncMock()
    process = AsyncMock()
    process.wait = AsyncMock(return_value=AsyncMock(
        exit_code=0,
        stdout="test output",
        stderr="",
        duration=1.0,
        memory_usage=100
    ))
    sandbox.process.start = AsyncMock(return_value=process)
    return sandbox


@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return Task(
        id="test-task-1",
        type=TaskType.ANALYSIS,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
        config=TaskConfig(),
        input_data={
            "code": "print('hello world')"
        }
    )


@pytest.mark.asyncio
async def test_agent_initialization(agent_config, mock_sandbox):
    """Test agent initialization."""
    with patch("clay_ai.agents.executor.Sandbox") as mock_sandbox_class:
        mock_sandbox_class.create = AsyncMock(return_value=mock_sandbox)
        
        agent = ExecutorAgent("test-agent", agent_config)
        await agent.initialize()
        
        assert agent.status == AgentStatus.IDLE
        assert agent.sandbox is not None
        mock_sandbox_class.create.assert_called_once()


@pytest.mark.asyncio
async def test_agent_cleanup(agent_config, mock_sandbox):
    """Test agent cleanup."""
    agent = ExecutorAgent("test-agent", agent_config)
    agent.sandbox = mock_sandbox
    
    await agent.cleanup()
    
    assert agent.status == AgentStatus.TERMINATED
    mock_sandbox.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_analysis_task(agent_config, mock_sandbox, sample_task):
    """Test execution of analysis task."""
    with patch("clay_ai.agents.executor.Sandbox") as mock_sandbox_class:
        mock_sandbox_class.create = AsyncMock(return_value=mock_sandbox)
        
        agent = ExecutorAgent("test-agent", agent_config)
        await agent.initialize()
        
        result = await agent.execute_task(sample_task)
        
        assert result.success is True
        assert "test output" in result.output["stdout"]
        assert result.metrics["memory_usage"] == 100
        mock_sandbox.process.start.assert_called_with(
            "python3",
            "-c",
            sample_task.input_data["code"]
        )


@pytest.mark.asyncio
async def test_execute_command_task(agent_config, mock_sandbox):
    """Test execution of command task."""
    with patch("clay_ai.agents.executor.Sandbox") as mock_sandbox_class:
        mock_sandbox_class.create = AsyncMock(return_value=mock_sandbox)
        
        agent = ExecutorAgent("test-agent", agent_config)
        await agent.initialize()
        
        task = Task(
            id="test-task-2",
            type=TaskType.EXECUTION,
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            config=TaskConfig(),
            input_data={
                "command": "ls -la"
            }
        )
        
        result = await agent.execute_task(task)
        
        assert result.success is True
        assert result.metrics["execution_time"] == 1.0
        mock_sandbox.process.start.assert_called_with(task.input_data["command"])


@pytest.mark.asyncio
async def test_unsupported_task_type(agent_config, mock_sandbox):
    """Test handling of unsupported task type."""
    agent = ExecutorAgent("test-agent", agent_config)
    agent.sandbox = mock_sandbox
    
    task = Task(
        id="test-task-3",
        type=TaskType.PLANNING,  # Unsupported type
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
        config=TaskConfig(),
        input_data={}
    )
    
    result = await agent.execute_task(task)
    
    assert result.success is False
    assert "unsupported task type" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_input_data(agent_config, mock_sandbox):
    """Test handling of missing input data."""
    agent = ExecutorAgent("test-agent", agent_config)
    agent.sandbox = mock_sandbox
    
    # Test missing code for analysis task
    analysis_task = Task(
        id="test-task-4",
        type=TaskType.ANALYSIS,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
        config=TaskConfig(),
        input_data={}
    )
    
    result = await agent.execute_task(analysis_task)
    assert result.success is False
    assert "no code provided" in result.error.lower()
    
    # Test missing command for execution task
    execution_task = Task(
        id="test-task-5",
        type=TaskType.EXECUTION,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
        config=TaskConfig(),
        input_data={}
    )
    
    result = await agent.execute_task(execution_task)
    assert result.success is False
    assert "no command provided" in result.error.lower() 