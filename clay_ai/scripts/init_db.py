"""Database initialization script for Clay AI."""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import init_db
from models.memory import Base


async def init_database():
    """Initialize database tables."""
    try:
        await init_db()
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(init_database())
