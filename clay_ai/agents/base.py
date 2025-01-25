"""
Base agent class for Clay AI framework.
"""

from typing import Optional
from abc import ABC, abstractmethod

from clay_ai.models.agents import AgentConfig
from clay_ai.models.tasks import Task, TaskResult


class BaseAgent(ABC):
    """Base class for all agents in the framework."""
    
    def __init__(
        self,
        name: str,
        role: str = "executor",
        config: Optional[AgentConfig] = None
    ) -> None:
        """Initialize base agent.
        
        Args:
            name: Agent name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ll
            role: Agent role (default: executor)
            config: Optional agent configuration
        """
        self.name = name
        self.role = role
        self.config = config or AgentConfig()
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize agent resources."""
        if not self.initialized:
            self.initialized = True
    
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if self.initialized:
            self.initialized = False 