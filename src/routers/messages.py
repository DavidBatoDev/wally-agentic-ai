# backend/src/routers/messages.py
"""
Router for message-related endpoints with improved alignment to the orchestrator.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, Optional, List
from uuid import UUID
import json

from ..dependencies.auth import get_current_user
from ..models.user import User
from ..models.conversation import TextMessageCreate, ActionMessageCreate, ButtonsMessage, InputsMessage
from ..utils.db_client import supabase_client
from ..llm.orchestrator import agent_orchestrator

router = APIRouter(
    prefix="/messages",
    tags=["messages"],
)


@router.post("/text", response_model=Dict[str, Any])
async def send_text_message(
    message: TextMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a text message and get a response from the agent.
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
            body=message.body,
        )
        
        # Process the message with the agent orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(message.conversation_id),
        }
        
        agent_response = await agent_orchestrator.process_user_message(
            conversation_id=message.conversation_id,
            user_message=message.body,
            context=context,
        )
        
        # Get the message ID from the agent response if available
        assistant_message_id = agent_response.get("message_id")
        assistant_message = None
        
        # If the orchestrator created a message and returned its ID, retrieve it
        if assistant_message_id:
            # The orchestrator has already created the message, just retrieve it
            recent_messages = supabase_client.get_conversation_messages(
                conversation_id=str(message.conversation_id),
                limit=10,  # Get more messages to ensure we find the right one
                offset=0,
            )
            
            # Find the message with the matching ID
            for msg in recent_messages:
                if msg.get("id") == assistant_message_id:
                    assistant_message = msg
                    break
        
        # If no assistant message was found (orchestrator didn't create one or we couldn't find it),
        # create one based on the response
        if not assistant_message:
            # Handle the response based on response_type
            response_type = agent_response.get("response_type", "text")
            response_body = agent_response.get("response", "I'm sorry, I couldn't process your request.")
            
            # Special handling for different response types
            if response_type == "buttons":
                buttons_data = ButtonsMessage(
                    prompt=agent_response.get("prompt", "Choose an option:"),
                    buttons=agent_response.get("buttons", [])
                )
                response_kind = "buttons"
                response_body = buttons_data.model_dump_json()
            
            elif response_type == "inputs":
                inputs_data = InputsMessage(
                    prompt=agent_response.get("prompt", "Please provide the following information:"),
                    inputs=agent_response.get("inputs", [])
                )
                response_kind = "inputs"
                response_body = inputs_data.model_dump_json()
            
            elif response_type == "file_card":
                response_kind = "file_card"
                response_body = json.dumps(agent_response.get("file_data", {}))
            
            elif response_type == "tool_result":
                # For tool results, we just use text kind but include the tool result in the response
                response_kind = "text"
                # The orchestrator already generates a user-friendly response
            
            else:
                # Default to text for any other response type
                response_kind = "text"
            
            # Create the assistant message
            assistant_message = supabase_client.create_message(
                conversation_id=str(message.conversation_id),
                sender="assistant",
                kind=response_kind,
                body=response_body,
            )
        
        return {
            "success": True,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "response_type": agent_response.get("response_type", "text"),
            "tool_result": agent_response.get("tool_result") if agent_response.get("response_type") == "tool_result" else None,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )


@router.post("/action", response_model=Dict[str, Any])
async def send_action_message(
    action: ActionMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send an action message (response to buttons or inputs).
    """
    try:
        # Get the conversation to verify the user has access
        conversation = supabase_client.get_conversation(str(action.conversation_id))
        
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
        
        # Store the action message
        action_data = {
            "action": action.action,
            "values": action.values or {},
        }
        user_message = supabase_client.create_message(
            conversation_id=str(action.conversation_id),
            sender="user",
            kind="action",
            body=json.dumps(action_data),  # Properly serialize to JSON
        )
        
        # Process the action with the agent orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(action.conversation_id),
        }
        
        # Check if orchestrator has a dedicated process_action method, otherwise use process_user_message
        if hasattr(agent_orchestrator, "process_action"):
            agent_response = await agent_orchestrator.process_action(
                conversation_id=action.conversation_id,
                action=action.action,
                values=action.values or {},
                context=context,
            )
        else:
            # Fallback to process_user_message with formatted action
            action_message = f"Action: {action.action}, Values: {json.dumps(action.values or {})}"
            agent_response = await agent_orchestrator.process_user_message(
                conversation_id=action.conversation_id,
                user_message=action_message,
                context={**context, "is_action": True, "action": action.action, "values": action.values or {}}
            )
        
        # Get the message ID from the agent response if available
        assistant_message_id = agent_response.get("message_id")
        assistant_message = None
        
        # If the orchestrator created a message and returned its ID, retrieve it
        if assistant_message_id:
            # The orchestrator has already created the message, just retrieve it
            recent_messages = supabase_client.get_conversation_messages(
                conversation_id=str(action.conversation_id),
                limit=10,  # Get more messages to ensure we find the right one
                offset=0,
            )
            
            # Find the message with the matching ID
            for msg in recent_messages:
                if msg.get("id") == assistant_message_id:
                    assistant_message = msg
                    break
        
        # If no assistant message was found (orchestrator didn't create one or we couldn't find it),
        # create one based on the response
        if not assistant_message:
            # Handle the response based on response_type
            response_type = agent_response.get("response_type", "text")
            response_body = agent_response.get("response", "I'm processing your action.")
            
            # Special handling for different response types
            if response_type == "buttons":
                buttons_data = ButtonsMessage(
                    prompt=agent_response.get("prompt", "Choose an option:"),
                    buttons=agent_response.get("buttons", [])
                )
                response_kind = "buttons"
                response_body = buttons_data.model_dump_json()
            
            elif response_type == "inputs":
                inputs_data = InputsMessage(
                    prompt=agent_response.get("prompt", "Please provide the following information:"),
                    inputs=agent_response.get("inputs", [])
                )
                response_kind = "inputs"
                response_body = inputs_data.model_dump_json()
            
            elif response_type == "file_card":
                response_kind = "file_card"
                response_body = json.dumps(agent_response.get("file_data", {}))
            
            elif response_type == "tool_result":
                # For tool results, we just use text kind but include the tool result in the response
                response_kind = "text"
                # The orchestrator already generates a user-friendly response
            
            else:
                # Default to text for any other response type
                response_kind = "text"
            
            # Create the assistant message
            assistant_message = supabase_client.create_message(
                conversation_id=str(action.conversation_id),
                sender="assistant",
                kind=response_kind,
                body=response_body,
            )
        
        return {
            "success": True,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "response_type": agent_response.get("response_type", "text"),
            "tool_result": agent_response.get("tool_result") if agent_response.get("response_type") == "tool_result" else None,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process action: {str(e)}",
        )


@router.get("/{conversation_id}", response_model=Dict[str, Any])
async def get_messages(
    conversation_id: UUID,
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