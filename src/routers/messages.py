"""
Enhanced router for message-related endpoints, adapted for the
new LangGraph orchestrator return-value (Realtime first, API second).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from pydantic import BaseModel, UUID4
import json

from ..dependencies import get_current_user, get_langgraph_orchestrator
from ..models.user import User
from ..db.db_client import supabase_client

router = APIRouter()


# ────────────────────────────────────────────────── payload models
class TextMessageCreate(BaseModel):
    conversation_id: UUID4
    body: str


class ActionMessageCreate(BaseModel):
    conversation_id: UUID4
    action: str
    values: Dict[str, Any] = {}
    source_message_id: UUID4 | None = None


class WorkflowStatusResponse(BaseModel):
    conversation_id: UUID4
    workflow_status: str
    current_step: str | None = None
    steps_completed: int = 0
    total_steps: int = 0
    user_confirmation_pending: bool = False


# ────────────────────────────────────────────────── helpers
def _guard_membership(conversation_id: str, current_user: User):
    convo = supabase_client.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if convo.get("profile_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this conversation",
        )
    return convo


# ────────────────────────────────────────────────── routes
@router.post("/text", response_model=Dict[str, Any])
async def send_text_message(
    message: TextMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Insert a user text message, hand it to the LangGraph orchestrator,
    and return a lightweight status object.

    NOTE: All assistant bubbles are pushed via Supabase Realtime, so the
    REST response no longer contains `assistant_message` / `response`.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        convo = _guard_membership(str(message.conversation_id), current_user)

        # ── persist the user message ------------------------------------------
        user_msg = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="user",
            kind="text",
            body=json.dumps({"text": message.body}),
        )

        # ── run the LangGraph orchestrator -------------------------------------
        orchestrator = get_langgraph_orchestrator()
        ctx = {
            "user_id": current_user.id,
            "conversation_id": str(message.conversation_id),
        }
        result = await orchestrator.process_user_message(
            conversation_id=str(message.conversation_id),
            user_message=message.body,
            context=ctx,
        )

        # optional: set conversation title on first turn
        if convo.get("title") in {None, "New Conversation"}:
            title = (message.body[:47] + "...") if len(message.body) > 50 else message.body
            supabase_client.update_conversation(
                conversation_id=str(message.conversation_id),
                data={"title": title},
            )

        # ── thin REST response --------------------------------------------------
        return {
            "success": True,
            "user_message": user_msg,
            # orchestrator now returns a lightweight summary:
            "inserted": result.get("inserted", 0),
            "workflow_status": result.get("workflow_status", "completed"),
            "steps_completed": result.get("steps_completed", 1),
        }

    except HTTPException:
        raise
    except Exception as exc:
        # log locally, then surface a friendly bubble to the user
        print(f"Error processing message: {exc}")

        error_payload = {
            "kind": "text",
            "text": (
                "I'm sorry, I encountered an error while processing your "
                "message. Please try again."
            ),
        }
        err_msg = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="assistant",
            kind="text",
            body=json.dumps(error_payload),
        )

        return {
            "success": False,
            "error": str(exc),
            "assistant_message": err_msg,
            "response": error_payload,
        }


@router.post("/action", response_model=Dict[str, Any])
async def handle_user_action(
    action_message: ActionMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Handle button-clicks, form submissions, etc.

    `handle_user_action` still expects the richer orchestrator return, so
    no change is required here.
    """
    try:
        _ = _guard_membership(str(action_message.conversation_id), current_user)

        # store the action as a user message
        user_action_msg = supabase_client.create_message(
            conversation_id=str(action_message.conversation_id),
            sender="user",
            kind="action",
            body=json.dumps(
                {
                    "action": action_message.action,
                    "values": action_message.values,
                    "source_message_id": (
                        str(action_message.source_message_id)
                        if action_message.source_message_id
                        else None
                    ),
                }
            ),
        )

        orchestrator = get_langgraph_orchestrator()
        result = await orchestrator.handle_user_action(
            conversation_id=str(action_message.conversation_id),
            action=action_message.action,
            values=action_message.values,
        )

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "response": result["response"],
            }

        return {
            "success": True,
            "user_action_message": user_action_msg,
            "assistant_message": result["message"],
            "response": result["response"],
            "workflow_status": result.get("workflow_status", "completed"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error handling user action: {exc}")
        error_payload = {
            "kind": "text",
            "text": "I'm sorry, I couldn't process your action at this time.",
        }
        return {
            "success": False,
            "error": str(exc),
            "response": error_payload,
        }


@router.get("/{conversation_id}", response_model=Dict[str, Any])
async def get_messages(
    conversation_id: UUID4,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return a page of messages for the given conversation."""
    try:
        _ = _guard_membership(str(conversation_id), current_user)

        messages = supabase_client.get_conversation_messages(
            conversation_id=str(conversation_id),
            limit=limit,
            offset=offset,
        )

        return {"success": True, "messages": messages}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {exc}",
        )
