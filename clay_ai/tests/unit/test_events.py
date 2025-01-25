"""Unit tests for event handling system."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import WebSocket, WebSocketDisconnect

from clay_ai.api.events import websocket_endpoint, broadcast_event
from clay_ai.core.events import EventManager


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    websocket = AsyncMock(spec=WebSocket)
    websocket.receive_json = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_event_manager():
    """Create a mock event manager."""
    return AsyncMock(spec=EventManager)


@pytest.mark.asyncio
async def test_websocket_connection(mock_websocket):
    """Test WebSocket connection establishment."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        client_id = "test-client"
        mock_websocket.receive_json.side_effect = WebSocketDisconnect()
        
        await websocket_endpoint(mock_websocket, client_id)
        
        mock_manager.connect.assert_called_once_with(client_id, mock_websocket)
        mock_manager.disconnect.assert_called_once_with(client_id, mock_websocket)


@pytest.mark.asyncio
async def test_websocket_event_handling(mock_websocket):
    """Test WebSocket event handling."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        client_id = "test-client"
        event_data = {
            "event": "test_event",
            "data": {"message": "hello"}
        }
        mock_websocket.receive_json.side_effect = [
            event_data,
            WebSocketDisconnect()
        ]
        
        await websocket_endpoint(mock_websocket, client_id)
        
        mock_manager.handle_event.assert_called_once_with(
            event_data["event"],
            event_data["data"]
        )
        mock_websocket.send_json.assert_called_with({
            "event": "test_event_ack",
            "data": {"status": "received"}
        })


@pytest.mark.asyncio
async def test_websocket_missing_event(mock_websocket):
    """Test handling of messages without event type."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        client_id = "test-client"
        event_data = {"data": {"message": "hello"}}
        mock_websocket.receive_json.side_effect = [
            event_data,
            WebSocketDisconnect()
        ]
        
        await websocket_endpoint(mock_websocket, client_id)
        
        mock_manager.handle_event.assert_not_called()
        mock_websocket.send_json.assert_called_with({
            "error": "Missing event type"
        })


@pytest.mark.asyncio
async def test_broadcast_to_all():
    """Test broadcasting event to all clients."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        event_data = {
            "event": "test_event",
            "data": {"message": "hello"}
        }
        
        response = await broadcast_event(event_data)
        
        mock_manager.broadcast_to_all.assert_called_once_with(
            event_data["event"],
            event_data["data"]
        )
        assert response["success"] is True
        assert "broadcasted successfully" in response["message"].lower()


@pytest.mark.asyncio
async def test_broadcast_to_client():
    """Test broadcasting event to specific client."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        client_id = "test-client"
        event_data = {
            "event": "test_event",
            "data": {"message": "hello"}
        }
        
        response = await broadcast_event(event_data, client_id)
        
        mock_manager.broadcast_to_client.assert_called_once_with(
            client_id,
            event_data["event"],
            event_data["data"]
        )
        assert response["success"] is True
        assert "broadcasted successfully" in response["message"].lower()


@pytest.mark.asyncio
async def test_broadcast_missing_event():
    """Test broadcasting without event type."""
    event_data = {"data": {"message": "hello"}}
    
    response = await broadcast_event(event_data)
    
    assert response["success"] is False
    assert "missing event type" in response["error"].lower()


@pytest.mark.asyncio
async def test_broadcast_error_handling():
    """Test error handling during broadcast."""
    with patch("clay_ai.api.events.event_manager") as mock_manager:
        mock_manager.broadcast_to_all.side_effect = Exception("Broadcast failed")
        event_data = {
            "event": "test_event",
            "data": {"message": "hello"}
        }
        
        response = await broadcast_event(event_data)
        
        assert response["success"] is False
        assert "broadcast failed" in response["error"].lower() 