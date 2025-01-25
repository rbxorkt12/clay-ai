"""
API endpoints for bulk linter error fixing.
Provides routes for fixing linter errors across multiple files.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException

from clay_ai.agents.linter import LinterAgent
from clay_ai.models.linter import BulkLinterResult
from clay_ai.models.agents import AgentConfig
from clay_ai.models.tasks import Task, TaskResult

router = APIRouter(prefix="/api/v1/linter")
linter_agent = LinterAgent(
    name="linter",
    config=AgentConfig(
        max_retries=3,
        timeout=300,
        memory_limit=512
    )
)


@router.post("/fix-bulk")
async def fix_errors_in_bulk(
    files: List[str],
    error_types: Optional[List[str]] = None
) -> BulkLinterResult:
    """Fix linter errors across multiple files.
    
    Args:
        files: List of file paths to process
        error_types: Optional list of error codes to fix
            (e.g., ["E501", "F401"])
        
    Returns:
        BulkLinterResult with statistics and fixes
    """
    try:
        await linter_agent.initialize()
        task = Task(
            type="linter",
            input={
                "files": files,
                "error_types": error_types
            }
        )
        result = await linter_agent.execute_task(task)
        await linter_agent.cleanup()
        return BulkLinterResult(**result.output)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(f"Error fixing linter errors: {e}")
        ) 