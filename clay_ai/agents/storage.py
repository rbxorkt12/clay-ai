"""Storage agent implementation for Clay AI."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import aiofiles
import os

from agents.base import BaseAgent
from models.agents import AgentConfig, AgentStatus
from models.tasks import Task, TaskResult, TaskType
from core.database import get_db_session


class StorageAgent(BaseAgent):
    """Agent responsible for storing and managing crawled data."""

    def __init__(
        self,
        name: str,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
    ) -> None:
        """Initialize storage agent."""
        super().__init__(name, config)
        self.storage_stats: Dict[str, Any] = {
            "total_files_stored": 0,
            "total_data_size": 0,
            "successful_stores": 0,
            "failed_stores": 0,
        }
        if isinstance(config, dict):
            self.storage_dir = config.get("parameters", {}).get(
                "storage_dir", "data/crawled"
            )
        else:
            self.storage_dir = (
                config.parameters.get("storage_dir", "data/crawled")
                if config
                else "data/crawled"
            )

    async def initialize(self) -> None:
        """Initialize storage resources."""
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_dir, exist_ok=True)
            self.status = AgentStatus.IDLE

            # Initialize base agent first
            await super().initialize()

            # Store initialization in memory
            async with get_db_session() as session:
                await self.add_memory(
                    content={
                        "event": "initialization",
                        "agent_type": "storage",
                        "storage_dir": self.storage_dir,
                    },
                    context="system",
                    importance=0.8,
                )
        except Exception as e:
            self.status = AgentStatus.ERROR
            # Store error in memory
            async with get_db_session() as session:
                await self.add_memory(
                    content={"event": "initialization_error", "error": str(e)},
                    context="error",
                    importance=1.0,
                )
            raise RuntimeError(f"Failed to initialize storage: {str(e)}")

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute storage task."""
        try:
            if task.type != TaskType.STORE:
                raise ValueError(f"Unsupported task type: {task.type}")

            # Store task start in memory
            await self.add_memory(
                content={
                    "event": "store_start",
                    "task_id": str(task.id),
                    "data_source": task.input.get("source_url", "unknown"),
                },
                context="task",
                importance=0.6,
            )

            # Store the data
            store_result = await self._store_data(
                data=task.input.get("data", {}),
                metadata=task.input.get("metadata", {}),
                filename=task.input.get("filename"),
            )

            # Update storage stats
            self.storage_stats["total_files_stored"] += 1
            if store_result["success"]:
                self.storage_stats["successful_stores"] += 1
                self.storage_stats["total_data_size"] += store_result[
                    "file_size"
                ]
            else:
                self.storage_stats["failed_stores"] += 1

            # Store successful storage in memory
            await self.add_memory(
                content={
                    "event": "store_complete",
                    "task_id": str(task.id),
                    "filename": store_result["filename"],
                    "success": store_result["success"],
                    "file_size": store_result["file_size"],
                },
                context="storage",
                importance=0.7,
            )

            return TaskResult(
                success=store_result["success"],
                error=store_result.get("error"),
                output={
                    "filename": store_result["filename"],
                    "file_path": store_result["file_path"],
                    "file_size": store_result["file_size"],
                },
                metrics=self.storage_stats,
            )

        except Exception as e:
            # Store error in memory
            await self.add_memory(
                content={
                    "event": "store_error",
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
                metrics=self.storage_stats,
            )

    async def _store_data(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
        filename: str = None,
    ) -> Dict[str, Any]:
        """Store data with metadata.

        Args:
            data: Data to store
            metadata: Additional metadata
            filename: Optional filename (generated if not provided)

        Returns:
            Dictionary containing storage results
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"crawled_data_{timestamp}.json"

            # Prepare full file path
            file_path = os.path.join(self.storage_dir, filename)

            # Prepare data with metadata
            full_data = {
                "metadata": {
                    **metadata,
                    "stored_at": datetime.utcnow().isoformat(),
                    "filename": filename,
                },
                "data": data,
            }

            # Write data to file
            async with aiofiles.open(file_path, mode="w") as f:
                await f.write(json.dumps(full_data, indent=2))

            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "filename": filename,
                "file_path": file_path,
                "file_size": file_size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filename": filename,
                "file_path": None,
                "file_size": 0,
            }
