# backend/src/routers/enhanced_messages.py
"""
Enhanced router for message-related endpoints with LangGraph orchestrator support.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from pydantic import BaseModel, UUID4
import json

from ..dependencies import get_current_user, get_langgraph_orchestrator
from ..models.user import User
from ..db.db_client import supabase_client

router = APIRouter()

class TextMessageCreate(BaseModel):
    conversation_id: UUID4
    body: str

class ActionMessageCreate(BaseModel):
    conversation_id: UUID4
    action: str
    values: Dict[str, Any] = {}
    source_message_id: UUID4 = None

class WorkflowStatusResponse(BaseModel):
    conversation_id: UUID4
    workflow_status: str
    current_step: str = None
    steps_completed: int = 0
    total_steps: int = 0
    user_confirmation_pending: bool = False

@router.post("/text", response_model=Dict[str, Any])
async def send_text_message(
    message: TextMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a text message and get a response from the enhanced LangGraph agent.
    """
    try:
        # Get the conversation to verify the user has access
        conversation = supabase_client.get_conversation(str(message.conversation_id))
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        
        # Check if user has access to this conversation
        if conversation.get("profile_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )
        
        # Store the user message
        user_message = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="user",
            kind="text",
            body=json.dumps({"text": message.body}),
        )
        
        # Get the LangGraph orchestrator from dependencies
        orchestrator = get_langgraph_orchestrator()
        
        # Process the message with the LangGraph orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(message.conversation_id),
        }
        
        agent_response = await orchestrator.process_user_message(
            conversation_id=str(message.conversation_id),
            user_message=message.body,
            context=context,
        )
        
        # Update conversation metadata if needed
        if conversation.get("title") is None or conversation.get("title") == "New Conversation":
            title = message.body[:50] + "..." if len(message.body) > 50 else message.body
            supabase_client.update_conversation(
                conversation_id=str(message.conversation_id),
                data={"title": title}
            )
        
        return {
            "success": True,
            "user_message": user_message,
            "assistant_message": agent_response["message"],
            "response": agent_response["response"],
            "workflow_status": agent_response.get("workflow_status", "completed"),
            "steps_completed": agent_response.get("steps_completed", 1),
            "user_confirmation_pending": agent_response.get("user_confirmation_pending", False)
        }
        
    except Exception as e:
        print(f"Error processing message: {e}")
        
        # Create an error message
        error_response = {
            "kind": "text",
            "text": "I'm sorry, I encountered an error while processing your message. Please try again."
        }
        
        error_message = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="assistant",
            kind="text",
            body=json.dumps(error_response)
        )
        
        return {
            "success": False,
            "error": str(e),
            "assistant_message": error_message,
            "response": error_response
        }

@router.post("/action", response_model=Dict[str, Any])
async def handle_user_action(
    action_message: ActionMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Handle user actions (like button clicks) for ongoing workflows.
    """
    try:
        # Get the conversation to verify the user has access
        conversation = supabase_client.get_conversation(str(action_message.conversation_id))
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        
        # Check if user has access to this conversation
        if conversation.get("profile_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )
        
        # Store the user action message
        user_action_message = supabase_client.create_message(
            conversation_id=str(action_message.conversation_id),
            sender="user",
            kind="action",
            body=json.dumps({
                "action": action_message.action,
                "values": action_message.values,
                "source_message_id": str(action_message.source_message_id) if action_message.source_message_id else None
            }),
        )
        
        # Get the LangGraph orchestrator
        orchestrator = get_langgraph_orchestrator()
        
        # Handle the user action
        response = await orchestrator.handle_user_action(
            conversation_id=str(action_message.conversation_id),
            action=action_message.action,
            values=action_message.values
        )
        
        if "error" in response:
            return {
                "success": False,
                "error": response["error"],
                "response": response["response"]
            }
        
        return {
            "success": True,
            "user_action_message": user_action_message,
            "assistant_message": response["message"],
            "response": response["response"],
            "workflow_status": response.get("workflow_status", "completed")
        }
        
    except Exception as e:
        print(f"Error handling user action: {e}")
        
        error_response = {
            "kind": "text",
            "text": "I'm sorry, I couldn't process your action at this time."
        }
        
        return {
            "success": False,
            "error": str(e),
            "response": error_response
        }

@router.get("/{conversation_id}/workflow-status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    conversation_id: UUID4,
    current_user: User = Depends(get_current_user),
) -> WorkflowStatusResponse:
    """
    Get the current workflow status for a conversation.
    """
    try:
        # ... existing validation code ...
        
        # Get the orchestrator to check workflow state
        orchestrator = get_langgraph_orchestrator()
        
        # Get the actual workflow state from LangGraph
        # This is a simplified version - you might want to implement 
        # a proper state query method in your orchestrator
        try:
            # You can add a method to your orchestrator to get current state
            # For now, return a basic status
            return WorkflowStatusResponse(
                conversation_id=conversation_id,
                workflow_status="completed",
                steps_completed=0,
                total_steps=0,
                user_confirmation_pending=False
            )
        except Exception as state_error:
            print(f"Error getting workflow state: {state_error}")
            return WorkflowStatusResponse(
                conversation_id=conversation_id,
                workflow_status="unknown",
                steps_completed=0,
                total_steps=0,
                user_confirmation_pending=False
            )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}",
        )
    
    
@router.get("/{conversation_id}", response_model=Dict[str, Any])
async def get_messages(
    conversation_id: UUID4,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get messages from a conversation.
    """
    try:
        # Get the conversation to verify the user has access
        conversation = supabase_client.get_conversation(str(conversation_id))
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        
        # Check if user has access to this conversation
        if conversation.get("profile_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )
        
        messages = supabase_client.get_conversation_messages(
            conversation_id=str(conversation_id),
            limit=limit,
            offset=offset,
        )
        
        return {
            "success": True,
            "messages": messages,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}",
        )