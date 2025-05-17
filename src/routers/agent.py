# backend/src/routers/agent.py
"""
Router for agent-related endpoints.
Handles LLM orchestration and tool dispatching.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from ..dependencies.auth import get_current_user
from ..models.user import User
from ..models.conversation import ConversationCreate, TextMessageCreate, Conversation
from ..llm.orchestrator import agent_orchestrator
from ..utils.db_client import supabase_client

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new conversation."""
    try:
        # Generate a new UUID for the conversation
        conversation_id = uuid4()
        
        # Create conversation in database
        conversation = supabase_client.create_conversation(
            conversation_id=str(conversation_id),
            profile_id=current_user.id,
            title=conversation_data.title,
        )
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "conversation": conversation,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}",
        )


@router.get("/conversations", response_model=Dict[str, Any])
async def list_conversations(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all conversations for the current user."""
    try:
        conversations = supabase_client.get_conversations(profile_id=current_user.id)
        
        return {
            "success": True,
            "conversations": conversations,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}",
        )


@router.get("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get a specific conversation."""
    try:
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
        
        messages = supabase_client.get_conversation_messages(str(conversation_id))
        
        return {
            "success": True,
            "conversation": conversation,
            "messages": messages,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}",
        )


@router.post("/message", response_model=Dict[str, Any])
async def send_message(
    message: TextMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send a message to the agent and get a response.
    
    This endpoint handles:
    1. Storing the user message
    2. Processing it through the LLM orchestrator
    3. Executing any required tools
    4. Returning the assistant's response
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
        }
        
        agent_response = await agent_orchestrator.process_user_message(
            conversation_id=message.conversation_id,
            user_message=message.body,
            context=context,
        )
        
        # Store the assistant's response
        assistant_message = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="assistant",
            kind="text",
            body=agent_response.get("response", "I'm sorry, I couldn't process your request."),
        )
        
        # If there's a tool result, also save that as metadata
        if agent_response.get("response_type") == "tool_result":
            tool_result = {
                "tool_name": agent_response.get("tool_name"),
                "tool_input": agent_response.get("tool_input"),
                "tool_result": agent_response.get("tool_result"),
            }
            
            # In a real implementation, you would save this to a separate table or as message metadata
        
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


@router.post("/langchain", response_model=Dict[str, Any])
async def process_with_langchain(
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Process a message using the LangChain agent framework.
    This is an alternative to the custom agent implementation.
    """
    try:
        conversation_id = UUID(data.get("conversation_id"))
        user_message = data.get("message")
        
        if not conversation_id or not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="conversation_id and message are required",
            )
        
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
        
        # Store the user message
        user_message_obj = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="user",
            kind="text",
            body=user_message,
        )
        
        # Process with LangChain
        context = {
            "user_id": current_user.id,
            **data.get("context", {}),
        }
        
        langchain_response = await agent_orchestrator.process_with_langchain(
            conversation_id=conversation_id,
            user_message=user_message,
            context=context,
        )
        
        # Store the assistant's response
        assistant_message = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="assistant",
            kind="text",
            body=langchain_response.get("response", "I'm sorry, I couldn't process your request."),
        )
        
        return {
            "success": True,
            "user_message": user_message_obj,
            "assistant_message": assistant_message,
            "response_type": langchain_response.get("response_type", "text"),
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process with LangChain: {str(e)}",
        )