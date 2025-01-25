# Clay AI

A Python Agent Framework for Building Production-Grade AI Applications

## Features

- **PydanticAI Integration**: Built on top of PydanticAI for robust model validation and structured responses
- **Type-Safe**: Leverages Python's type system with Pydantic V2 for maximum reliability
- **Async-First**: Built with asyncio and FastAPI for high-performance async operations
- **Sandboxed Execution**: Secure code execution using E2B sandboxes
- **Monitoring & Observability**: Prometheus metrics and OpenTelemetry integration
- **Production-Ready**: Includes error handling, logging, and security best practices

## Project Structure

```
clay_ai/
├── core/                    # Core functionality and services
│   ├── __init__.py
│   ├── config.py           # Configuration management using Pydantic V2
│   ├── database.py         # Database connections and session management
│   ├── events.py           # Event system and WebSocket management
│   ├── logging.py          # Logging configuration and utilities
│   └── security.py         # Authentication and authorization
│
├── models/                  # Pydantic V2 models and schemas
│   ├── __init__.py
│   ├── agents.py           # Agent-related models and enums
│   ├── tasks.py            # Task-related models and enums
│   ├── events.py           # Event-related models and schemas
│   ├── responses.py        # Common response models
│   └── validators.py       # Custom validators and field types
│
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── base.py             # Base agent class with PydanticAI integration
│   ├── executor.py         # Task execution agent
│   ├── planner.py          # Task planning agent
│   ├── observer.py         # Monitoring agent
│   └── coordinator.py      # Multi-agent coordination
│
├── api/                     # FastAPI routes and endpoints
│   ├── __init__.py
│   ├── agents.py           # Agent management endpoints
│   ├── tasks.py            # Task management endpoints
│   ├── events.py           # WebSocket and event endpoints
│   ├── auth.py             # Authentication endpoints
│   └── monitoring.py       # Health and metrics endpoints
│
├── utils/                   # Utility functions and helpers
│   ├── __init__.py
│   ├── async_utils.py      # Async helper functions
│   ├── sandbox.py          # E2B sandbox utilities
│   ├── metrics.py          # Prometheus metrics helpers
│   └── telemetry.py        # OpenTelemetry utilities
│
└── tests/                   # Test suite
    ├── unit/               # Unit tests
    │   ├── __init__.py
    │   ├── test_agents.py  # Agent tests
    │   ├── test_tasks.py   # Task tests
    │   ├── test_events.py  # Event system tests
    │   └── test_models.py  # Model validation tests
    │
    └── integration/        # Integration tests
        ├── __init__.py
        ├── test_api.py     # API endpoint tests
        ├── test_workflows.py# End-to-end workflow tests
        └── test_sandbox.py # E2B sandbox tests
```

## Dependencies

### Runtime Dependencies

- fastapi: ^0.104.1
- uvicorn: ^0.24.0
- pydantic: ^2.5.3
- pydantic-ai: ^0.0.19
- pydantic-settings: ^2.1.0
- sqlalchemy: ^2.0.25
- asyncpg: ^0.28.0
- redis: ^5.0.1
- e2b: ^0.13.8
- prometheus-client: ^0.19.0
- opentelemetry-api: ^1.21.0

### Development Dependencies

- pytest: ^7.4.4
- pytest-asyncio: ^0.23.3
- black: ^23.12.1
- mypy: ^1.8.0
- ruff: ^0.1.11

## Installation

```bash
# Using pip
pip install clay-ai

# Using Poetry (recommended)
poetry add clay-ai
```

## Quick Start

```python
from clay_ai import Agent, Task
from clay_ai.models import TaskType, TaskPriority

# Create an agent
agent = Agent(name="analysis-agent", capabilities=["python", "data-analysis"])

# Create and execute a task
task = Task(
    type=TaskType.ANALYSIS,
    priority=TaskPriority.HIGH,
    input_data={"code": "print('Hello from Clay AI!')"}
)

# Execute task
result = await agent.execute_task(task)
print(result.output)
```

## Development

1. Clone the repository
2. Install dependencies:

   ```bash
   poetry install
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   ```

4. Run tests:

   ```bash
   poetry run pytest
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.
