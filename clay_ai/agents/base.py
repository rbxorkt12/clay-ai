"""
Base agent class for Clay AI framework.
"""

from typing import Dict, Any, Optional, List
from uuid import uuid4
from datetime import datetime
import json

from models.memory import AgentMemory
from models.agents import AgentDB, AgentConfig
from core.database import get_db_session


class BaseAgent:
    """Base agent class with memory capabilities."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.id = str(uuid4())
        self.name = name
        self.config = config or {}
        self.memory = AgentMemory()
        self.status = "initialized"
        self.current_task = None
        self.error = None
        self.metrics = {}

    def _serialize_config(self) -> str:
        """Serialize agent configuration to JSON string."""
        if isinstance(self.config, dict):
            return json.dumps(self.config)
        elif isinstance(self.config, AgentConfig):
            return json.dumps(self.config.model_dump())
        else:
            return json.dumps({})

    async def initialize(self):
        async with get_db_session() as session:
            try:
                # Create agent DB entry first
                agent_db = AgentDB(
                    id=self.id,
                    name=self.name,
                    config=self._serialize_config(),
                    status=self.status,
                    metrics=json.dumps(self.metrics),
                )
                session.add(agent_db)
                await session.flush()  # Ensure the agent is saved before adding memories

                # Now initialize memory
                self.memory.add(
                    content={
                        "event": "initialization",
                        "agent_type": self.__class__.__name__.lower(),
                        "capabilities": self.get_capabilities(),
                    },
                    context="system",
                    importance=0.8,
                )
                await self.memory.save_to_db(session, self.id)
                await session.commit()  # Commit all changes together

            except Exception as e:
                await session.rollback()  # Rollback on error
                self.error = str(e)
                raise RuntimeError(f"Failed to initialize agent: {str(e)}")

    async def cleanup(self):
        """Clean up agent resources."""
        try:
            # Save any remaining memories to database
            async with get_db_session() as session:
                await self.memory.save_to_db(session, self.id)
                await session.commit()
        except Exception as e:
            self.error = str(e)
            print(f"Error during memory cleanup: {str(e)}")
        finally:
            # Clear memory references
            if hasattr(self, "memory"):
                self.memory.short_term.clear()
                self.memory.long_term.clear()

    async def add_memory(
        self,
        content: Dict[str, Any],
        context: str = "general",
        importance: float = 0.5,
        session=None,
    ):
        try:
            self.memory.add(content, context, importance)
            if session:
                await self.memory.save_to_db(session, self.id)
            else:
                async with get_db_session() as new_session:
                    await self.memory.save_to_db(new_session, self.id)
        except Exception as e:
            print(f"Error adding memory: {str(e)}")
            raise

    def get_capabilities(self) -> List[Dict[str, Any]]:
        return []

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Subclasses must implement execute_task method"
        )
