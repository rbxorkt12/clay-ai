"""Coordinator agent implementation for Clay AI."""

from typing import Dict, Any, List
from datetime import datetime

from agents.base import BaseAgent
from models.agents import AgentConfig, AgentStatus
from models.tasks import Task, TaskResult, TaskType


class CoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating multiple agents and tasks."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        """Initialize coordinator agent."""
        super().__init__(name, config)
        self.coordination_stats: Dict[str, Any] = {
            "total_coordinations": 0,
            "active_agents": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
        }
        self.active_workflows: Dict[str, List[str]] = (
            {}
        )  # workflow_id -> task_ids

    async def initialize(self) -> None:
        """Initialize coordinator resources."""
        try:
            self.status = AgentStatus.IDLE
            # Store initialization in memory
            await self.add_memory(
                content={
                    "event": "initialization",
                    "agent_type": "coordinator",
                    "coordination_start": datetime.utcnow().isoformat(),
                },
                context="system",
                importance=0.8,
            )
        except Exception as e:
            self.status = AgentStatus.ERROR
            # Store error in memory
            await self.add_memory(
                content={"event": "initialization_error", "error": str(e)},
                context="error",
                importance=1.0,
            )
            raise RuntimeError(f"Failed to initialize coordinator: {str(e)}")

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute coordination task."""
        try:
            if task.type != TaskType.COORDINATION:
                raise ValueError(f"Unsupported task type: {task.type}")

            # Store task start in memory
            await self.add_memory(
                content={
                    "event": "coordination_start",
                    "task_id": str(task.id),
                    "workflow": task.input.get("workflow_id"),
                },
                context="task",
                importance=0.6,
            )

            # Coordinate workflow execution
            workflow_result = await self._coordinate_workflow(task)

            # Update coordination stats
            self.coordination_stats["total_coordinations"] += 1
            if workflow_result["success"]:
                self.coordination_stats["completed_workflows"] += 1
            else:
                self.coordination_stats["failed_workflows"] += 1

            # Store workflow completion in memory
            await self.add_memory(
                content={
                    "event": "workflow_complete",
                    "task_id": str(task.id),
                    "workflow_id": task.input.get("workflow_id"),
                    "success": workflow_result["success"],
                    "num_tasks": len(
                        workflow_result.get("completed_tasks", [])
                    ),
                    "execution_time": workflow_result.get("execution_time", 0),
                },
                context="workflow",
                importance=0.7,
            )

            # Store any workflow failures with high importance
            if not workflow_result["success"]:
                await self.add_memory(
                    content={
                        "event": "workflow_failure",
                        "task_id": str(task.id),
                        "workflow_id": task.input.get("workflow_id"),
                        "error": workflow_result.get("error"),
                        "failed_tasks": workflow_result.get(
                            "failed_tasks", []
                        ),
                    },
                    context="error",
                    importance=0.9,
                )

            return TaskResult(
                success=workflow_result["success"],
                error=workflow_result.get("error"),
                output=workflow_result,
                metrics=self.coordination_stats,
            )

        except Exception as e:
            # Store error in memory
            await self.add_memory(
                content={
                    "event": "coordination_error",
                    "task_id": str(task.id),
                    "error": str(e),
                },
                context="error",
                importance=0.9,
            )

            return TaskResult(
                success=False,
                error=str(e),
                output={},
                metrics=self.coordination_stats,
            )

    async def _coordinate_workflow(self, task: Task) -> Dict[str, Any]:
        """Coordinate workflow execution.

        This is a simplified implementation. In a real system,
        this would handle complex workflow orchestration.
        """
        workflow_id = task.input.get("workflow_id")
        if not workflow_id:
            raise ValueError("No workflow_id provided")

        start_time = datetime.utcnow()

        try:
            # Example workflow coordination
            workflow_tasks = task.input.get("tasks", [])
            completed_tasks = []
            failed_tasks = []

            # Track workflow tasks
            self.active_workflows[workflow_id] = [
                str(t.get("id")) for t in workflow_tasks
            ]
            self.coordination_stats["active_agents"] = len(workflow_tasks)

            # Store workflow tracking in memory
            await self.add_memory(
                content={
                    "event": "workflow_tracking",
                    "workflow_id": workflow_id,
                    "num_tasks": len(workflow_tasks),
                    "task_types": [t.get("type") for t in workflow_tasks],
                },
                context="tracking",
                importance=0.8,
            )

            # Simulate task execution
            for task_info in workflow_tasks:
                if task_info.get("success", True):
                    completed_tasks.append(task_info)
                else:
                    failed_tasks.append(task_info)

                # Store task completion in memory
                await self.add_memory(
                    content={
                        "event": "task_complete",
                        "workflow_id": workflow_id,
                        "task_id": task_info.get("id"),
                        "success": task_info.get("success", True),
                    },
                    context="task",
                    importance=0.6,
                )

            # Cleanup workflow tracking
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                "success": len(failed_tasks) == 0,
                "workflow_id": workflow_id,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "execution_time": execution_time,
            }

        except Exception as e:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            raise e
