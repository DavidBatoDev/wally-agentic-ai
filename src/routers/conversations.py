# backend/src/routers/conversations.py
"""
Router for conversation management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from ..dependencies.auth import get_current_user
from ..models.user import User
from ..models.conversation import ConversationCreate, Conversation, Message
from ..utils.db_client import supabase_client

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.post("/", response_model=Dict[str, Any])
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a new conversation."""
    try:
        # Create conversation in database
        conversation = supabase_client.create_conversation(
            profile_id=current_user.id,
            title=conversation_data.title,
        )
        
        return {
            "success": True,
            "conversation_id": conversation["id"],
            "conversation": conversation,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}",
        )


@router.get("/", response_model=Dict[str, Any])
async def list_conversations(
    limit: int = 10,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all conversations for the current user."""
    try:
        conversations = supabase_client.get_user_conversations(
            profile_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "conversations": conversations,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}",
        )


@router.get("/{conversation_id}", response_model=Dict[str, Any])
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


@router.delete("/{conversation_id}", response_model=Dict[str, Any])
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a conversation."""
    try:
        # First get the conversation to verify ownership
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
        
        # Instead of deleting, we'll update is_active to False
        # This is a soft delete that preserves the data
        updated = supabase_client.client.table("conversations").update({
            "is_active": False
        }).eq("id", str(conversation_id)).execute()
        
        return {
            "success": True,
            "message": "Conversation archived successfully",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}",
        )