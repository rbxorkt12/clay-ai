"""Event handling and WebSocket management for Clay AI."""

from typing import Dict, Set, Callable, Awaitable, Any
from fastapi import WebSocket
from redis.asyncio import Redis

from clay_ai.core.config import settings


class EventManager:
    """Manages WebSocket connections and event broadcasting."""
    
    def __init__(self) -> None:
        """Initialize event manager."""
        self.active_connections = {}  # Dict[str, Set[WebSocket]]
        self.redis = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
        self.event_handlers = {}  # Dict[str, Callable[[Any], Awaitable[None]]]
    
    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Register new WebSocket connection."""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
    
    async def disconnect(self, client_id: str, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        self.active_connections[client_id].remove(websocket)
        if not self.active_connections[client_id]:
            del self.active_connections[client_id]
    
    async def broadcast_to_client(
        self,
        client_id: str,
        event: str,
        data: Any
    ) -> None:
        """Broadcast event to specific client's connections."""
        if client_id in self.active_connections:
            message = {"event": event, "data": data}
            for connection in self.active_connections[client_id]:
                await connection.send_json(message)
    
    async def broadcast_to_all(self, event: str, data: Any) -> None:
        """Broadcast event to all active connections."""
        message = {"event": event, "data": data}
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_json(message)
    
    def register_handler(
        self,
        event: str,
        handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """Register event handler."""
        self.event_handlers[event] = handler
    
    async def handle_event(self, event: str, data: Any) -> None:
        """Handle incoming event."""
        if event in self.event_handlers:
            await self.event_handlers[event](data)


event_manager = EventManager() 