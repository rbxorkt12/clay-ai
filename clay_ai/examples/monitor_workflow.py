"""Example of a system monitoring workflow using Clay AI agents."""

import asyncio
from uuid import uuid4
import psutil
import time

from models.agents import AgentConfig, AgentRole, AgentCapability
from models.tasks import Task, TaskType
from agents.observer import ObserverAgent
from agents.executor import ExecutorAgent
from agents.coordinator import CoordinatorAgent


async def collect_system_metrics() -> dict:
    """Collect current system metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "network_io": psutil.net_io_counters()._asdict(),
    }


async def run_monitor_workflow():
    """Run a system monitoring workflow."""

    # Initialize agents
    observer = ObserverAgent(
        name="system_monitor",
        config=AgentConfig(
            role=AgentRole.OBSERVER,
            capabilities=[
                AgentCapability(
                    name="system_monitoring",
                    version="1.0",
                    parameters={
                        "metrics": ["cpu", "memory", "disk", "network"],
                        "thresholds": {
                            "cpu_percent": 80.0,
                            "memory_percent": 85.0,
                            "disk_usage": 90.0,
                        },
                    },
                )
            ],
        ),
    )

    executor = ExecutorAgent(
        name="action_executor",
        config=AgentConfig(
            role=AgentRole.EXECUTOR,
            capabilities=[
                AgentCapability(
                    name="system_actions",
                    version="1.0",
                    parameters={
                        "allowed_actions": ["cleanup", "alert", "log"]
                    },
                )
            ],
        ),
    )

    coordinator = CoordinatorAgent(
        name="workflow_coordinator",
        config=AgentConfig(
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability(name="workflow_management", version="1.0")
            ],
        ),
    )

    # Initialize all agents
    await asyncio.gather(
        observer.initialize(),
        executor.initialize(),
        coordinator.initialize(),
    )

    try:
        # Monitoring loop
        for _ in range(3):  # Monitor for 3 iterations
            # Collect system metrics
            metrics = await collect_system_metrics()

            # Create observation task
            observe_task = Task(
                id=uuid4(),
                type=TaskType.OBSERVATION,
                input={
                    "target": "system",
                    "metrics": metrics,
                    "thresholds": observer.config.capabilities[0].parameters[
                        "thresholds"
                    ],
                },
            )

            # Execute observation
            observe_result = await observer.execute_task(observe_task)

            if observe_result.success:
                # Check for anomalies
                anomalies = observe_result.output.get("anomalies", [])
                if anomalies:
                    print(f"Detected anomalies: {anomalies}")

                    # Create execution task for each anomaly
                    for anomaly in anomalies:
                        execute_task = Task(
                            id=uuid4(),
                            type=TaskType.EXECUTION,
                            input={
                                "action": "alert",
                                "data": {
                                    "anomaly_type": anomaly["type"],
                                    "value": anomaly["value"],
                                    "timestamp": anomaly["timestamp"],
                                },
                            },
                        )

                        # Execute action
                        execute_result = await executor.execute_task(
                            execute_task
                        )

                        if execute_result.success:
                            print(f"Action executed: {execute_result.output}")
                        else:
                            print(f"Action failed: {execute_result.error}")
                else:
                    print("System status: Normal")

                # Print monitoring stats
                print(f"Monitoring stats: {observe_result.metrics}")
                print(
                    f"System health score: {observe_result.output['system_health']}"
                )
            else:
                print(f"Observation failed: {observe_result.error}")

            # Wait before next iteration
            await asyncio.sleep(5)

    finally:
        # Cleanup
        await asyncio.gather(
            observer.cleanup(),
            executor.cleanup(),
            coordinator.cleanup(),
        )


if __name__ == "__main__":
    asyncio.run(run_monitor_workflow())
