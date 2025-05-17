# backend/src/routers/upload.py
"""
Router for file upload endpoints.
Handles document and image uploads to Supabase Storage.
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import os
import mimetypes
from datetime import datetime

from src.dependencies.auth import get_current_user
from src.models.user import User
from src.models.document import UploadResponse, FileObjectCreate
from utils.db_client import supabase_client
from src.utils.storage import upload_file_to_storage
from src.config import get_settings

settings = get_settings()
router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)


@router.post("/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: UUID = Form(...),
    document_type: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
) -> UploadResponse:
    """
    Upload a file (image or document) to the system.
    
    This endpoint:
    1. Validates the file type
    2. Uploads it to Supabase Storage
    3. Creates metadata record in the database
    4. Returns file information for further processing
    """
    try:
        # Check if conversation exists and user has access
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
        
        # Validate file type
        content_type = file.content_type
        if not content_type:
            # Try to guess from filename if content_type is missing
            content_type, _ = mimetypes.guess_type(file.filename)
        
        # List of allowed file types
        allowed_types = [
            "image/jpeg", "image/png", "image/webp", 
            "application/pdf", 
            "application/msword", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {content_type} not allowed. Allowed types: {', '.join(allowed_types)}",
            )
        
        # Generate file ID and prepare path
        file_id = uuid4()
        file_extension = os.path.splitext(file.filename)[1] if "." in file.filename else ""
        object_key = f"{current_user.id}/{file_id}{file_extension}"
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Upload to Supabase Storage
        upload_result = await upload_file_to_storage(
            bucket_name=settings.STORAGE_BUCKET_NAME,
            object_key=object_key,
            file_content=file_content,
            content_type=content_type,
        )
        
        # Create file object in database
        file_object = FileObjectCreate(
            profile_id=current_user.id,
            bucket=settings.STORAGE_BUCKET_NAME,
            object_key=object_key,
            mime_type=content_type,
            size_bytes=file_size,
        )
        
        db_file = supabase_client.create_file_object(file_object)
        
        # Create a file message in the conversation
        file_message = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="user",
            kind="file",
            body=f"Uploaded file: {file.filename}",
            metadata={
                "file_id": str(file_id),
                "filename": file.filename,
                "content_type": content_type,
                "document_type": document_type,
            },
        )
        
        # Return upload response
        return UploadResponse(
            file_id=file_id,
            conversation_id=conversation_id,
            object_key=object_key,
            mime_type=content_type,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.get("/files", response_model=Dict[str, Any])
async def list_user_files(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all files uploaded by the current user."""
    try:
        files = supabase_client.get_user_files(profile_id=current_user.id)
        
        return {
            "success": True,
            "files": files,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {str(e)}",
        )


@router.get("/files/{file_id}", response_model=Dict[str, Any])
async def get_file_info(
    file_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get information about a specific file."""
    try:
        file_info = supabase_client.get_file_info(file_id=str(file_id))
        
        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found",
            )
        
        # Check if user has access to this file
        if file_info.get("profile_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this file",
            )
        
        return {
            "success": True,
            "file_info": file_info,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get file info: {str(e)}",
        )


@router.delete("/files/{file_id}", response_model=Dict[str, Any])
async def delete_file(
    file_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a specific file."""
    try:
        file_info = supabase_client.get_file_info(file_id=str(file_id))
        
        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found",
            )
        
        # Check if user has access to this file
        if file_info.get("profile_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this file",
            )
        
        # Delete from storage
        await supabase_client.delete_file_from_storage(
            bucket_name=file_info.get("bucket"),
            object_key=file_info.get("object_key"),
        )
        
        # Delete from database
        supabase_client.delete_file_object(file_id=str(file_id))
        
        return {
            "success": True,
            "message": "File deleted successfully",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )