"""WebSocket endpoints for event handling."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from clay_ai.core.events import event_manager

router = APIRouter(prefix="/events", tags=["events"])


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    """WebSocket endpoint for real-time event handling."""
    try:
        await event_manager.connect(client_id, websocket)
        while True:
            try:
                # Receive and process messages
                data = await websocket.receive_json()
                event = data.get("event")
                payload = data.get("data", {})
                
                if not event:
                    await websocket.send_json({
                        "error": "Missing event type"
                    })
                    continue
                
                # Handle the event
                await event_manager.handle_event(event, payload)
                
                # Send acknowledgment
                await websocket.send_json({
                    "event": f"{event}_ack",
                    "data": {"status": "received"}
                })
                
            except WebSocketDisconnect:
                await event_manager.disconnect(client_id, websocket)
                break
            except Exception as e:
                await websocket.send_json({
                    "error": str(e)
                })
    except Exception as e:
        # Handle connection errors
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass  # Connection might already be closed


@router.post("/broadcast", tags=["events"])
async def broadcast_event(
    event_data: Dict[str, Any],
    client_id: Optional[str] = None
) -> Dict[str, Any]:
    """Broadcast event to all clients or specific client."""
    try:
        event = event_data.get("event")
        data = event_data.get("data", {})
        
        if not event:
            return {
                "success": False,
                "error": "Missing event type"
            }
        
        if client_id:
            await event_manager.broadcast_to_client(client_id, event, data)
        else:
            await event_manager.broadcast_to_all(event, data)
        
        return {
            "success": True,
            "message": "Event broadcasted successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 