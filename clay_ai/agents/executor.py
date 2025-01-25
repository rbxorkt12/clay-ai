"""Executor agent implementation for Clay AI."""

import time
from typing import Dict, Any
from e2b import Sandbox

from clay_ai.agents.base import BaseAgent
from clay_ai.models.agents import AgentConfig, AgentStatus
from clay_ai.models.tasks import Task, TaskResult, TaskType
from clay_ai.core.config import settings


class ExecutorAgent(BaseAgent):
    """Agent responsible for executing tasks in sandboxed environments."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        """Initialize executor agent."""
        super().__init__(name, config)
        self.sandbox = None  # type: Sandbox | None
        self.execution_stats: Dict[str, Any] = {
            "total_execution_time": 0,
            "avg_memory_usage": 0,
            "successful_executions": 0,
            "failed_executions": 0
        }

    async def initialize(self) -> None:
        """Initialize sandbox environment."""
        try:
            self.sandbox = await Sandbox.create(
                api_key=settings.E2B_API_KEY,
                template="python3",
                timeout=self.config.timeout
            )
            self.status = AgentStatus.IDLE
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise RuntimeError(f"Failed to initialize sandbox: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self.sandbox:
            await self.sandbox.close()
        self.status = AgentStatus.TERMINATED

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute assigned task in sandbox environment."""
        start_time = time.time()
        try:
            if task.type not in [TaskType.ANALYSIS, TaskType.EXECUTION]:
                raise ValueError(f"Unsupported task type: {task.type}")

            # Execute task based on type
            if task.type == TaskType.ANALYSIS:
                result = await self._execute_analysis(task)
            else:
                result = await self._execute_command(task)

            # Update execution stats
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            self.execution_stats["successful_executions"] += 1
            self.execution_stats["avg_memory_usage"] = (
                self.execution_stats["avg_memory_usage"] +
                result.metrics.get("memory_usage", 0)
            ) / 2

            return result

        except Exception as e:
            self.execution_stats["failed_executions"] += 1
            return TaskResult(
                success=False,
                error=str(e),
                output={},
                metrics={
                    "execution_time": time.time() - start_time,
                    "memory_usage": 0
                }
            )

    async def _execute_analysis(self, task: Task) -> TaskResult:
        """Execute analysis task."""
        try:
            # Prepare analysis code
            code = task.input_data.get("code", "")
            if not code:
                raise ValueError("No code provided for analysis")

            # Execute code in sandbox
            process = await self.sandbox.process.start("python3", "-c", code)
            output = await process.wait()

            return TaskResult(
                success=output.exit_code == 0,
                error=None,
                output={
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "exit_code": output.exit_code
                },
                metrics={
                    "execution_time": output.duration,
                    "memory_usage": output.memory_usage
                }
            )

        except Exception as e:
            raise RuntimeError(f"Analysis execution failed: {str(e)}")

    async def _execute_command(self, task: Task) -> TaskResult:
        """Execute command task."""
        try:
            # Get command from task input
            command = task.input_data.get("command", "")
            if not command:
                raise ValueError("No command provided for execution")

            # Execute command in sandbox
            process = await self.sandbox.process.start(command)
            output = await process.wait()

            return TaskResult(
                success=output.exit_code == 0,
                error=None,
                output={
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "exit_code": output.exit_code
                },
                metrics={
                    "execution_time": output.duration,
                    "memory_usage": output.memory_usage
                }
            )

        except Exception as e:
            raise RuntimeError(f"Command execution failed: {str(e)}") 