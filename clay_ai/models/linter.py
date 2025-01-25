"""
Models for linter error detection and fixing.
Uses Pydantic V2 for strong typing and validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class LinterError(BaseModel):
    """Represents a single linter error."""
    file_path: str = Field(
        ...,
        description="Path to the file containing the error"
    )
    line_number: int = Field(
        ...,
        description="Line number where error occurs"
    )
    column: Optional[int] = Field(
        None,
        description="Column number where error occurs"
    )
    error_code: str = Field(
        ...,
        description="Error code (e.g., E501, F401)"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    severity: int = Field(
        ...,
        description="Error severity (1: warning, 2: error)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "clay_ai/core/vercel_ai.py",
                "line_number": 3,
                "column": 85,
                "error_code": "E501",
                "message": "line too long (85 > 79 characters)",
                "severity": 1
            }
        }
    )


class LinterFix(BaseModel):
    """Represents a fix for a linter error."""
    error: LinterError = Field(
        ...,
        description="The error being fixed"
    )
    original_line: str = Field(
        ...,
        description="Original line of code"
    )
    fixed_line: str = Field(
        ...,
        description="Fixed line of code"
    )
    fix_description: str = Field(
        ...,
        description="Description of the fix applied"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "file_path": "clay_ai/core/vercel_ai.py",
                    "line_number": 3,
                    "column": 85,
                    "error_code": "E501",
                    "message": "line too long (85 > 79 characters)",
                    "severity": 1
                },
                "original_line": (
                    "Provides a bridge between E2B sandbox execution and "
                    "Vercel AI streaming capabilities."
                ),
                "fixed_line": (
                    "Provides a bridge between E2B sandbox execution and "
                    "Vercel AI\nstreaming capabilities."
                ),
                "fix_description": "Split long line into multiple lines"
            }
        }
    )


class BulkLinterResult(BaseModel):
    """Results from bulk linter error fixing."""
    total_files: int = Field(
        ...,
        description="Total number of files processed"
    )
    total_errors: int = Field(
        ...,
        description="Total number of errors found"
    )
    fixed_errors: int = Field(
        ...,
        description="Number of errors successfully fixed"
    )
    failed_fixes: int = Field(
        ...,
        description="Number of errors that couldn't be fixed"
    )
    fixes: List[LinterFix] = Field(
        default_factory=list,
        description="List of applied fixes"
    )
    remaining_errors: List[LinterError] = Field(
        default_factory=list,
        description="List of unfixed errors"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_files": 3,
                "total_errors": 10,
                "fixed_errors": 8,
                "failed_fixes": 2,
                "fixes": [],
                "remaining_errors": []
            }
        }
    ) 