"""
Vercel AI integration module for Clay AI.
Provides a bridge between E2B sandbox execution and Vercel AI streaming capabilities.
"""

from typing import AsyncGenerator, Dict, Optional
from vercel_ai_core import Message, StreamingTextResponse
from vercel_ai_py import AIStream
from e2b import Sandbox

from clay_ai.core.config import settings
from clay_ai.models.agents import AgentResponse
from clay_ai.models.tasks import TaskResult


class VercelAIBridge:
    """Bridge between E2B sandbox execution and Vercel AI streaming."""
    
    def __init__(self) -> None:
        self.sandbox: Optional[Sandbox] = None
        
    async def initialize(self) -> None:
        """Initialize E2B sandbox with Vercel AI integration."""
        if not self.sandbox:
            self.sandbox = await Sandbox.create(
                template="python3",
                api_key=settings.E2B_API_KEY,
                enable_video=True
            )
    
    async def execute_with_streaming(
        self, 
        messages: list[Message],
        model_params: Optional[Dict] = None
    ) -> StreamingTextResponse:
        """Execute code in E2B sandbox and stream results via Vercel AI."""
        await self.initialize()
        
        async def stream_generator() -> AsyncGenerator[str, None]:
            # Process messages and prepare code
            code = self._prepare_code(messages)
            
            # Execute in sandbox
            result = await self.sandbox.run_python(code)
            
            # Stream results
            yield "Executing code...\n"
            if result.stdout:
                yield f"Output:\n{result.stdout}\n"
            if result.stderr:
                yield f"Errors:\n{result.stderr}\n"
                
            # Get video stream if available
            if self.sandbox.video_stream_url:
                yield f"\nVideo stream available at: {self.sandbox.video_stream_url}\n"
        
        return StreamingTextResponse(
            AIStream(stream_generator())
        )
    
    def _prepare_code(self, messages: list[Message]) -> str:
        """Extract and prepare code from messages."""
        # For now, just take the last user message
        last_msg = next(
            (msg for msg in reversed(messages) if msg["role"] == "user"),
            None
        )
        return last_msg["content"] if last_msg else ""
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.sandbox:
            await self.sandbox.close()
            self.sandbox = None


# Global instance for reuse
vercel_ai_bridge = VercelAIBridge() 