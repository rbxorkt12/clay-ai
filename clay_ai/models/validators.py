"""Custom validators and field types using PydanticAI."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import AgentValidator, ModelValidator, validate_llm_response

from clay_ai.core.config import settings


class CodeValidator(AgentValidator):
    """Validates Python code for security and syntax."""
    
    @field_validator("code")
    @classmethod
    def validate_python_code(cls, code: str) -> str:
        """Validate Python code for security and syntax."""
        if not code.strip():
            raise ValueError("Code cannot be empty")
        
        # Basic security checks
        forbidden = ["os.system", "subprocess", "eval", "exec"]
        for term in forbidden:
            if term in code:
                raise ValueError(f"Forbidden term '{term}' in code")
        
        return code


class PromptValidator(ModelValidator):
    """Validates and enhances prompts for LLM interactions."""
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        """Validate and enhance prompt for LLM."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Add system context if not present
        if not prompt.startswith("System:"):
            prompt = f"System: You are a Clay AI agent.\n\n{prompt}"
        
        return prompt


class ResponseValidator(ModelValidator):
    """Validates LLM responses for structure and content."""
    
    @validate_llm_response
    def validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM response structure and content."""
        required_fields = ["thought", "action", "output"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")
        
        return response


class MetricsValidator(BaseModel):
    """Validates agent performance metrics."""
    
    execution_time: float = Field(gt=0)
    memory_usage: float = Field(gt=0)
    cpu_usage: Optional[float] = Field(None, ge=0, le=100)
    error_rate: Optional[float] = Field(None, ge=0, le=1)
    
    @field_validator("execution_time")
    @classmethod
    def validate_execution_time(cls, value: float) -> float:
        """Validate execution time is within limits."""
        if value > settings.TASK_TIMEOUT:
            raise ValueError(f"Execution time exceeds timeout: {value}")
        return value 