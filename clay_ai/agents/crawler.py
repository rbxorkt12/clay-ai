"""Crawler agent implementation for Clay AI."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import json

from agents.base import BaseAgent
from models.agents import AgentConfig, AgentStatus, AgentRole, AgentCapability
from models.tasks import Task, TaskResult, TaskType


class CrawlerAgent(BaseAgent):
    """Agent responsible for web crawling tasks."""

    def __init__(
        self,
        name: str,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
    ) -> None:
        """Initialize crawler agent."""
        if config is None:
            config = AgentConfig(
                role=AgentRole.EXECUTOR,
                capabilities=[
                    AgentCapability(
                        name="web_crawling",
                        version="1.0",
                        parameters={"max_pages": 100, "timeout": 30},
                    )
                ],
                memory_limit=1024,
                priority=1,
                timeout=300,
                max_retries=3,
                parameters={},
            )
        elif isinstance(config, dict):
            config = AgentConfig(**config)
        super().__init__(name, config)
        self.crawling_stats = {
            "total_pages": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "total_data_size": 0,
            "response_time": 0.0,
        }
        self.session = None

    async def initialize(self) -> None:
        """Initialize the crawler agent."""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()

            # Initialize base agent
            await super().initialize()

            self.status = "ready"

        except Exception as e:
            self.status = "error"
            self.error = str(e)
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # First cleanup session if it exists and is not closed
            if (
                hasattr(self, "session")
                and self.session
                and not self.session.closed
            ):
                await self.session.close()

            # Then cleanup base agent resources
            await super().cleanup()

        except Exception as e:
            self.error = str(e)
            print(f"Error during cleanup: {str(e)}")
        finally:
            # Ensure session is properly cleaned up
            if hasattr(self, "session") and self.session:
                if (
                    hasattr(self.session, "connector")
                    and self.session.connector
                ):
                    self.session.connector._closed = True
                self.session = None

    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get agent capabilities."""
        return [
            {
                "name": "web_crawling",
                "version": "1.0",
                "parameters": {"max_retries": 3},
            }
        ]

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a crawling task."""
        try:
            if task.type != TaskType.CRAWL:
                raise ValueError(f"Unsupported task type: {task.type}")

            url = task.input.get("url")
            selectors = task.input.get("selectors", {})

            if not url:
                raise ValueError("URL is required for crawling task")

            start_time = datetime.utcnow()

            # Crawl the page
            async with self.session.get(url) as response:
                html = await response.text()

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Extract data using selectors
            data = {}
            for key, selector in selectors.items():
                elements = soup.select(selector)
                data[key] = [elem.get_text(strip=True) for elem in elements]

            # Update stats
            self.crawling_stats["total_pages"] += 1
            self.crawling_stats["successful_crawls"] += 1
            self.crawling_stats["total_data_size"] += len(html)
            self.crawling_stats["response_time"] += response_time

            # Add memory entry for successful crawl
            await self.add_memory(
                content={
                    "event": "crawl_success",
                    "url": url,
                    "data_size": len(html),
                    "response_time": response_time,
                },
                context="crawl",
                importance=0.7,
            )

            return TaskResult(
                success=True,
                output={
                    "url": url,
                    "data": data,
                    "stats": self.crawling_stats,
                },
            )

        except Exception as e:
            # Update stats for failed crawl
            self.crawling_stats["total_pages"] += 1
            self.crawling_stats["failed_crawls"] += 1

            # Add memory entry for failed crawl
            await self.add_memory(
                content={"event": "crawl_error", "url": url, "error": str(e)},
                context="error",
                importance=1.0,
            )

            return TaskResult(
                success=False,
                error=str(e),
                output={
                    "url": url,
                    "stats": self.crawling_stats,
                },
            )

    async def _crawl_page(
        self, url: str, selectors: Dict[str, str], max_retries: int = 3
    ) -> Dict[str, Any]:
        """Crawl a web page using provided selectors.

        Args:
            url: Target URL to crawl
            selectors: CSS selectors to extract data
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing crawl results
        """
        if not url:
            raise ValueError("No URL provided for crawling")

        start_time = datetime.utcnow()

        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Extract data using selectors
                        data = {}
                        for key, selector in selectors.items():
                            elements = soup.select(selector)
                            data[key] = [
                                el.get_text(strip=True) for el in elements
                            ]

                        response_time = (
                            datetime.utcnow() - start_time
                        ).total_seconds()

                        return {
                            "success": True,
                            "data": data,
                            "response_time": response_time,
                            "status_code": response.status,
                        }
                    else:
                        if attempt == max_retries - 1:
                            return {
                                "success": False,
                                "data": {},
                                "error": f"HTTP {response.status}",
                                "response_time": (
                                    datetime.utcnow() - start_time
                                ).total_seconds(),
                                "status_code": response.status,
                            }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "data": {},
                        "error": str(e),
                        "response_time": (
                            datetime.utcnow() - start_time
                        ).total_seconds(),
                        "status_code": None,
                    }

        return {
            "success": False,
            "data": {},
            "error": "Max retries exceeded",
            "response_time": (datetime.utcnow() - start_time).total_seconds(),
            "status_code": None,
        }
