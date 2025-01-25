"""Observer agent implementation for Clay AI."""

from typing import Dict, Any, List
from datetime import datetime

from agents.base import BaseAgent
from models.agents import AgentConfig, AgentStatus
from models.tasks import Task, TaskResult, TaskType


class ObserverAgent(BaseAgent):
    """Agent responsible for monitoring and logging system behavior."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        """Initialize observer agent."""
        super().__init__(name, config)
        self.observation_stats: Dict[str, Any] = {
            "total_observations": 0,
            "anomalies_detected": 0,
            "system_health_score": 1.0,
            "last_observation_time": None,
        }

    async def initialize(self) -> None:
        """Initialize observer resources."""
        try:
            self.status = AgentStatus.IDLE
            # Store initialization in memory
            await self.add_memory(
                content={
                    "event": "initialization",
                    "agent_type": "observer",
                    "monitoring_start": datetime.utcnow().isoformat(),
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
            raise RuntimeError(f"Failed to initialize observer: {str(e)}")

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute observation task."""
        try:
            if task.type != TaskType.OBSERVATION:
                raise ValueError(f"Unsupported task type: {task.type}")

            # Store task start in memory
            await self.add_memory(
                content={
                    "event": "observation_start",
                    "task_id": str(task.id),
                    "target": task.input.get("target", "system"),
                },
                context="task",
                importance=0.6,
            )

            # Perform observation
            observations = await self._collect_observations(task)
            anomalies = self._detect_anomalies(observations)

            # Update observation stats
            self.observation_stats["total_observations"] += 1
            self.observation_stats["anomalies_detected"] += len(anomalies)
            self.observation_stats["last_observation_time"] = (
                datetime.utcnow().isoformat()
            )

            # Update system health score
            if observations:
                health_impact = len(anomalies) / len(observations)
                self.observation_stats["system_health_score"] *= (
                    1 - health_impact
                )

            # Store observations in memory
            await self.add_memory(
                content={
                    "event": "observation_complete",
                    "task_id": str(task.id),
                    "num_observations": len(observations),
                    "num_anomalies": len(anomalies),
                    "health_score": self.observation_stats[
                        "system_health_score"
                    ],
                },
                context="observation",
                importance=0.7,
            )

            # Store any anomalies with high importance
            if anomalies:
                await self.add_memory(
                    content={
                        "event": "anomalies_detected",
                        "task_id": str(task.id),
                        "anomalies": anomalies,
                    },
                    context="anomaly",
                    importance=0.9,
                )

            return TaskResult(
                success=True,
                output={
                    "observations": observations,
                    "anomalies": anomalies,
                    "system_health": self.observation_stats[
                        "system_health_score"
                    ],
                },
                metrics=self.observation_stats,
            )

        except Exception as e:
            # Store error in memory
            await self.add_memory(
                content={
                    "event": "observation_error",
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
                metrics=self.observation_stats,
            )

    async def _collect_observations(self, task: Task) -> List[Dict[str, Any]]:
        """Collect system observations.

        This is a simplified implementation. In a real system,
        this would collect metrics, logs, and other system data.
        """
        target = task.input.get("target", "system")
        observations = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "target": target,
                "metrics": {
                    "cpu_usage": 0.5,  # Example metric
                    "memory_usage": 0.3,  # Example metric
                    "active_tasks": 5,  # Example metric
                },
            }
        ]

        return observations

    def _detect_anomalies(
        self, observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in observations.

        This is a simplified implementation. In a real system,
        this would use more sophisticated anomaly detection.
        """
        anomalies = []

        for obs in observations:
            metrics = obs.get("metrics", {})

            # Example anomaly detection rules
            if metrics.get("cpu_usage", 0) > 0.8:
                anomalies.append(
                    {
                        "type": "high_cpu_usage",
                        "value": metrics["cpu_usage"],
                        "timestamp": obs["timestamp"],
                    }
                )

            if metrics.get("memory_usage", 0) > 0.9:
                anomalies.append(
                    {
                        "type": "high_memory_usage",
                        "value": metrics["memory_usage"],
                        "timestamp": obs["timestamp"],
                    }
                )

        return anomalies
