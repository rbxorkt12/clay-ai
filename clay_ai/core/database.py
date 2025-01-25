"""Database connection and session management for Clay AI."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncContextManager
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base

from core.config import settings


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,  # Enable connection health checks
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Create declarative base for models
Base = declarative_base()


@asynccontextmanager
async def db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with transaction management."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with db_transaction() as session:
        yield session


async def init_db() -> None:
    """Initialize database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class DatabaseSession:
    """Database session context manager for better transaction management."""

    def __init__(self) -> None:
        """Initialize session manager."""
        self.session: AsyncSession = AsyncSessionLocal()

    async def __aenter__(self) -> AsyncSession:
        """Enter context and begin transaction."""
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and handle transaction."""
        try:
            if exc_type is not None:
                await self.session.rollback()
            else:
                await self.session.commit()
        finally:
            await self.session.close()


def get_db_session() -> AsyncContextManager[AsyncSession]:
    """Get database session with context management."""
    return DatabaseSession()
