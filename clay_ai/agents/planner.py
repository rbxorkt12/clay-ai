"""Planner agent implementation for Clay AI."""

from typing import List, Dict, Any
from uuid import uuid4

from agents.base import BaseAgent
from models.agents import AgentConfig, AgentStatus
from models.tasks import Task, TaskResult, TaskType


class PlannerAgent(BaseAgent):
    """Agent responsible for planning and breaking down complex tasks."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        """Initialize planner agent."""
        super().__init__(name, config)
        self.planning_stats: Dict[str, Any] = {
            "total_plans_created": 0,
            "avg_subtasks_per_plan": 0,
            "successful_plans": 0,
            "failed_plans": 0,
        }

    async def initialize(self) -> None:
        """Initialize planner resources."""
        try:
            self.status = AgentStatus.IDLE
            # Store initialization in memory
            await self.add_memory(
                content={
                    "event": "initialization",
                    "agent_type": "planner",
                    "capabilities": [
                        c.dict() for c in self.config.capabilities
                    ],
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
            raise RuntimeError(f"Failed to initialize planner: {str(e)}")

    async def execute_task(self, task: Task) -> TaskResult:
        """Plan and break down a complex task."""
        try:
            if task.type != TaskType.PLANNING:
                raise ValueError(f"Unsupported task type: {task.type}")

            # Store task start in memory
            await self.add_memory(
                content={
                    "event": "planning_start",
                    "task_id": str(task.id),
                    "input": task.input,
                },
                context="task",
                importance=0.6,
            )

            # Create subtasks based on task input
            subtasks = await self._create_subtasks(task)

            # Update planning stats
            self.planning_stats["total_plans_created"] += 1
            self.planning_stats["avg_subtasks_per_plan"] = (
                self.planning_stats["avg_subtasks_per_plan"] + len(subtasks)
            ) / 2
            self.planning_stats["successful_plans"] += 1

            # Store successful planning in memory
            await self.add_memory(
                content={
                    "event": "planning_complete",
                    "task_id": str(task.id),
                    "num_subtasks": len(subtasks),
                    "subtask_types": [st.type for st in subtasks],
                },
                context="task",
                importance=0.7,
            )

            return TaskResult(
                success=True,
                output={
                    "subtasks": [st.dict() for st in subtasks],
                    "execution_order": list(range(len(subtasks))),
                },
                metrics={
                    "num_subtasks": len(subtasks),
                    "planning_stats": self.planning_stats,
                },
            )

        except Exception as e:
            self.planning_stats["failed_plans"] += 1

            # Store error in memory
            await self.add_memory(
                content={
                    "event": "planning_error",
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
                metrics={"planning_stats": self.planning_stats},
            )

    async def _create_subtasks(self, task: Task) -> List[Task]:
        """Create subtasks based on the main task.

        This is a simplified implementation. In a real system,
        this would use LLM or other AI capabilities to break down tasks.
        """
        subtasks = []

        # Example: Break down into analysis and execution subtasks
        analysis_task = Task(
            id=uuid4(),
            type=TaskType.ANALYSIS,
            input={"code": task.input.get("code", "")},
            parent_task=task.id,
        )
        subtasks.append(analysis_task)

        execution_task = Task(
            id=uuid4(),
            type=TaskType.EXECUTION,
            input={"command": task.input.get("command", "")},
            parent_task=task.id,
        )
        subtasks.append(execution_task)

        # Store subtask creation in memory
        await self.add_memory(
            content={
                "event": "subtasks_created",
                "parent_task_id": str(task.id),
                "subtask_ids": [str(st.id) for st in subtasks],
            },
            context="planning",
            importance=0.8,
        )

        return subtasks
