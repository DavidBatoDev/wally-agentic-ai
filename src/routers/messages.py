# backend/src/routers/messages.py
"""
Router for message-related endpoints with improved alignment to the orchestrator.
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
from typing import Dict, Any
from pydantic import BaseModel, UUID4
import json
from uuid import UUID4 as UUIDv4

from ..dependencies import get_current_user, get_agent_orchestrator
from ..models.user import User
from ..db.db_client import supabase_client

router = APIRouter(
    prefix="/api/messages",
    tags=["messages"],
)

class TextMessageCreate(BaseModel):
    conversation_id: UUID4
    body: str

class ActionMessageCreate(BaseModel):
    conversation_id: UUID4
    action: str
    values: Dict[str, Any] = {}
    source_message_id: UUID4 = None

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
            body=json.dumps({"text": message.body}),  # Format as per our schema
        )
        
        # Get the agent orchestrator from dependencies
        agent_orchestrator = get_agent_orchestrator()
        
        # Process the message with the agent orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(message.conversation_id),
        }
        
        agent_response = await agent_orchestrator.process_user_message(
            conversation_id=str(message.conversation_id),
            user_message=message.body,
            context=context,
        )
        
        # Update conversation metadata (optional)
        # For example, update title if this is the first message
        if conversation.get("title") is None or conversation.get("title") == "New Conversation":
            # Generate a title based on the first message
            title = message.body[:50] + "..." if len(message.body) > 50 else message.body
            supabase_client.update_conversation(
                conversation_id=str(message.conversation_id),
                data={"title": title}
            )
        
        return {
            "success": True,
            "user_message": user_message,
            "assistant_message": agent_response["message"],
            "response": agent_response["response"]
        }
        
    except Exception as e:
        # Log the error
        print(f"Error processing message: {e}")
        
        # Create an error message
        error_message = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="assistant",
            kind="text",
            body=json.dumps({"text": "I'm sorry, I encountered an error while processing your message."})
        )
        
        # Return an error response
        return {
            "success": False,
            "error": str(e),
            "assistant_message": error_message,
            "response": {"kind": "text", "text": "I'm sorry, I encountered an error while processing your message."}
        }

@router.post("/action", response_model=Dict[str, Any])
async def send_action_message(
    message: ActionMessageCreate,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Send an action message (e.g., button click or form submit) and get a response.
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
        
        # Get the agent orchestrator from dependencies
        agent_orchestrator = get_agent_orchestrator()
        
        # Process the action with the agent orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(message.conversation_id),
        }
        
        agent_response = await agent_orchestrator.process_action(
            conversation_id=str(message.conversation_id),
            action=message.action,
            values=message.values,
            context=context,
        )
        
        return {
            "success": True,
            "assistant_message": agent_response["message"],
            "response": agent_response["response"]
        }
        
    except Exception as e:
        # Log the error
        print(f"Error processing action: {e}")
        
        # Create an error message
        error_message = supabase_client.create_message(
            conversation_id=str(message.conversation_id),
            sender="assistant",
            kind="text",
            body=json.dumps({"text": "I'm sorry, I encountered an error while processing your action."})
        )
        
        # Return an error response
        return {
            "success": False,
            "error": str(e),
            "assistant_message": error_message,
            "response": {"kind": "text", "text": "I'm sorry, I encountered an error while processing your action."}
        }

@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(
    conversation_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload a file to the conversation and get a response from the agent.
    """
    try:
        # Get the conversation to verify the user has access
        conversation = supabase_client.get_conversation(conversation_id)
        
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
        
        # Upload the file to storage
        file_content = await file.read()
        
        # Generate a unique file path
        file_id = str(UUIDv4())
        object_key = f"{current_user.id}/{conversation_id}/{file_id}_{file.filename}"
        
        # Upload to Supabase Storage
        supabase_client.upload_file(
            bucket="user_uploads",
            object_key=object_key,
            file_content=file_content,
            file_type=file.content_type
        )
        
        # Create a file object record
        file_object = supabase_client.create_file_object(
            profile_id=current_user.id,
            bucket="user_uploads",
            object_key=object_key,
            mime_type=file.content_type,
            size_bytes=len(file_content)
        )
        
        # Get file info for processing
        file_info = {
            "name": file.filename,
            "mime": file.content_type,
            "size": len(file_content),
            "bucket": "user_uploads",
            "object_key": object_key
        }
        
        # Get the agent orchestrator from dependencies
        agent_orchestrator = get_agent_orchestrator()
        
        # Process the file upload with the agent orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": conversation_id,
        }
        
        agent_response = await agent_orchestrator.process_file_upload(
            conversation_id=conversation_id,
            file_id=file_object["id"],
            file_info=file_info,
            context=context,
        )
        
        return {
            "success": True,
            "file_object": file_object,
            "assistant_message": agent_response["message"],
            "response": agent_response["response"]
        }
        
    except Exception as e:
        # Log the error
        print(f"Error processing file upload: {e}")
        
        # Create an error message
        error_message = supabase_client.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps({"text": "I'm sorry, I encountered an error while processing your file upload."})
        )
        
        # Return an error response
        return {
            "success": False,
            "error": str(e),
            "assistant_message": error_message,
            "response": {"kind": "text", "text": "I'm sorry, I encountered an error while processing your file upload."}
        }

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