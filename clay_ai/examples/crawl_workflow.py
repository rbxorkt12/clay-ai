"""Example of a web crawling workflow using Clay AI agents."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from agents.crawler import CrawlerAgent
from agents.storage import StorageAgent
from models.agents import AgentRole, AgentCapability
from models.tasks import Task, TaskType, TaskConfig


async def run_crawl_workflow():
    """Run a web crawling workflow."""
    crawler = None
    storage = None
    try:
        # Initialize agents
        crawler = CrawlerAgent(
            name="hacker_news_crawler",
            config={
                "role": AgentRole.EXECUTOR,
                "capabilities": [
                    {
                        "name": "web_crawling",
                        "version": "1.0",
                        "parameters": {"max_pages": 100, "timeout": 30},
                    }
                ],
                "memory_limit": 1024,
                "priority": 1,
                "timeout": 30,
                "max_retries": 3,
                "parameters": {"user_agent": "Clay AI Crawler/1.0"},
            },
        )

        storage = StorageAgent(
            name="file_storage",
            config={
                "role": AgentRole.EXECUTOR,
                "capabilities": [
                    {
                        "name": "file_storage",
                        "version": "1.0",
                        "parameters": {},
                    }
                ],
                "memory_limit": 1024,
                "priority": 1,
                "timeout": 30,
                "max_retries": 3,
                "parameters": {
                    "storage_dir": "data/crawled",
                    "format": "json",
                },
            },
        )

        # Initialize agents
        await asyncio.gather(crawler.initialize(), storage.initialize())

        # Create crawling task
        crawl_task = Task(
            type=TaskType.CRAWL,
            config=TaskConfig(timeout=30, max_retries=3),
            input={
                "url": "https://news.ycombinator.com",
                "selectors": {
                    "titles": ".titleline > a",
                    "scores": ".score",
                    "comments": ".subtext a:last-child",
                },
            },
        )

        # Execute crawling task
        crawl_result = await crawler.execute_task(crawl_task)

        if crawl_result.success:
            # Prepare data for storage
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"crawled_data_{timestamp}.json"

            storage_task = Task(
                type=TaskType.STORE,
                config=TaskConfig(timeout=30, max_retries=3),
                input={
                    "filename": filename,
                    "data": {
                        "metadata": {
                            "source_url": crawl_task.input["url"],
                            "crawl_timestamp": timestamp,
                            "crawl_stats": crawl_result.output["stats"],
                            "selectors": crawl_task.input["selectors"],
                        },
                        "data": crawl_result.output["data"],
                    },
                },
            )

            # Store crawled data
            storage_result = await storage.execute_task(storage_task)

            if storage_result.success:
                print(
                    f"Data stored successfully at {storage_result.output['file_path']}"
                )
                print("\nCrawling stats:")
                print(
                    f"Total pages crawled: {crawl_result.output['stats']['total_pages']}"
                )
                print(
                    f"Successful crawls: {crawl_result.output['stats']['successful_crawls']}"
                )
                print(
                    f"Failed crawls: {crawl_result.output['stats']['failed_crawls']}"
                )
                print(
                    f"Total data size: {crawl_result.output['stats']['total_data_size']} bytes"
                )
                print(
                    f"Response time: {crawl_result.output['stats']['response_time']:.6f} seconds"
                )
            else:
                print(f"Storage failed: {storage_result.error}")
        else:
            print(f"Crawling failed: {crawl_result.error}")

    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        raise

    finally:
        # Always cleanup agents
        if crawler or storage:
            try:
                cleanup_tasks = []
                if crawler:
                    cleanup_tasks.append(crawler.cleanup())
                if storage:
                    cleanup_tasks.append(storage.cleanup())
                if cleanup_tasks:
                    await asyncio.gather(*cleanup_tasks)
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    # Ensure data directory exists
    Path("data/crawled").mkdir(parents=True, exist_ok=True)

    # Run workflow
    asyncio.run(run_crawl_workflow())
