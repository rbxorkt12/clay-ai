"""
Linter agent for fixing code style and quality issues.
Uses black, isort, and ruff for automated fixes.
"""

from typing import List
from pathlib import Path
import subprocess
import black
import isort

from agents.base import BaseAgent
from models.tasks import Task, TaskResult, TaskType
from models.agents import AgentConfig
from models.linter import LinterError, LinterFix, BulkLinterResult


class LinterAgent(BaseAgent):
    """Agent for fixing linter errors using black, isort, and ruff."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        super().__init__(name=name, config=config, role="linter")
        self.black_mode = black.Mode()
        self.isort_config = isort.Config()
        self.ruff_path = "ruff"  # Use system installed ruff

    async def execute_task(self, task: Task) -> TaskResult:
        if task.type != TaskType.LINT:
            return TaskResult(
                success=False,
                error="Unsupported task type. Expected 'lint'",
                output={},
            )

        try:
            files = task.input.get("files", [])
            if not files:
                return TaskResult(
                    success=False,
                    error="No files provided for linting",
                    output={},
                )

            result = await self.fix_errors_in_bulk(files)
            return TaskResult(
                success=True, error=None, output=result.model_dump()
            )
        except Exception as e:
            error_msg = "Error executing linting task: " + str(e)
            return TaskResult(success=False, error=error_msg, output={})

    async def fix_errors_in_bulk(self, files: List[str]) -> BulkLinterResult:
        fixes: List[LinterFix] = []
        remaining_errors: List[LinterError] = []
        total_files = len(files)
        total_errors = 0
        fixed_errors = 0
        failed_fixes = 0

        for file_path in files:
            try:
                path = Path(file_path)
                if not path.exists():
                    remaining_errors.append(
                        LinterError(
                            file_path=file_path,
                            line_number=0,
                            column=0,
                            error_code="FILE_NOT_FOUND",
                            message="File not found",
                            severity=2,
                        )
                    )
                    total_errors += 1
                    failed_fixes += 1
                    continue

                # Apply black formatting
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    formatted = black.format_str(content, mode=self.black_mode)
                    with open(file_path, "w") as f:
                        f.write(formatted)
                    fixes.append(
                        LinterFix(
                            error=LinterError(
                                file_path=file_path,
                                line_number=0,
                                column=0,
                                error_code="BLACK_FORMAT",
                                message="Code formatted with black",
                                severity=1,
                            ),
                            original_line=content,
                            fixed_line=formatted,
                            fix_description="Applied black formatting",
                        )
                    )
                    fixed_errors += 1
                except Exception as e:
                    remaining_errors.append(
                        LinterError(
                            file_path=file_path,
                            line_number=0,
                            column=0,
                            error_code="BLACK_ERROR",
                            message=str(e),
                            severity=2,
                        )
                    )
                    total_errors += 1
                    failed_fixes += 1

                # Apply isort
                try:
                    isort.file(file_path, config=self.isort_config)
                    fixes.append(
                        LinterFix(
                            error=LinterError(
                                file_path=file_path,
                                line_number=0,
                                column=0,
                                error_code="ISORT_FORMAT",
                                message="Imports sorted with isort",
                                severity=1,
                            ),
                            original_line="",  # isort modifies file in place
                            fixed_line="",
                            fix_description="Sorted imports with isort",
                        )
                    )
                    fixed_errors += 1
                except Exception as e:
                    remaining_errors.append(
                        LinterError(
                            file_path=file_path,
                            line_number=0,
                            column=0,
                            error_code="ISORT_ERROR",
                            message=str(e),
                            severity=2,
                        )
                    )
                    total_errors += 1
                    failed_fixes += 1

                # Apply ruff using CLI
                try:
                    result = subprocess.run(
                        [self.ruff_path, "--fix", file_path],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        fixes.append(
                            LinterFix(
                                error=LinterError(
                                    file_path=file_path,
                                    line_number=0,
                                    column=0,
                                    error_code="RUFF_FORMAT",
                                    message="Fixed issues with ruff",
                                    severity=1,
                                ),
                                original_line="",  # ruff modifies file in place
                                fixed_line="",
                                fix_description="Applied ruff fixes",
                            )
                        )
                        fixed_errors += 1
                    else:
                        remaining_errors.append(
                            LinterError(
                                file_path=file_path,
                                line_number=0,
                                column=0,
                                error_code="RUFF_ERROR",
                                message=result.stderr,
                                severity=2,
                            )
                        )
                        total_errors += 1
                        failed_fixes += 1
                except Exception as e:
                    remaining_errors.append(
                        LinterError(
                            file_path=file_path,
                            line_number=0,
                            column=0,
                            error_code="RUFF_ERROR",
                            message=str(e),
                            severity=2,
                        )
                    )
                    total_errors += 1
                    failed_fixes += 1

            except Exception as e:
                remaining_errors.append(
                    LinterError(
                        file_path=file_path,
                        line_number=0,
                        column=0,
                        error_code="PROCESSING_ERROR",
                        message=str(e),
                        severity=2,
                    )
                )
                total_errors += 1
                failed_fixes += 1

        return BulkLinterResult(
            total_files=total_files,
            total_errors=total_errors,
            fixed_errors=fixed_errors,
            failed_fixes=failed_fixes,
            fixes=fixes,
            remaining_errors=remaining_errors,
        )
