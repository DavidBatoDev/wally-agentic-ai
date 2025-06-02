# backend/src/routers/uploads.py
"""
Router for file upload endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any
from pydantic import BaseModel, UUID4
import uuid
import os
import json

from ..dependencies import get_current_user, get_langgraph_orchestrator
from ..models.user import User
from ..db.db_client import supabase_client

router = APIRouter()

# Allowed file types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"
}
ALLOWED_PDF_TYPES = {"application/pdf"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES

# Max file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

class FileUploadResponse(BaseModel):
    success: bool
    file_id: str
    public_url: str
    message: Dict[str, Any]
    response: Dict[str, Any]
    mime_type: str
    size_bytes: int
    workflow_status: str
    steps_completed: int

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    
    # Check MIME type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file.content_type}' not allowed. Allowed types: images (JPEG, PNG, GIF, WebP, BMP, TIFF) and PDF"
        )

def generate_object_key(user_id: str, filename: str) -> str:
    """Generate a unique object key for the file."""
    # Create a unique filename to avoid conflicts
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Structure: user_id/year/month/unique_id_original_name
    from datetime import datetime
    now = datetime.now()
    year_month = now.strftime("%Y/%m")
    
    # Sanitize the original filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-").rstrip()
    
    return f"{user_id}/{year_month}/{file_id}_{safe_filename}"

@router.post("/file", response_model=FileUploadResponse)
async def upload_file(
    conversation_id: UUID4 = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> FileUploadResponse:
    """
    Upload a single image or PDF file and create a file message in the conversation.
    """
    try:
        # Validate the file
        validate_file(file)
        
        # Verify conversation exists and user has access
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
        
        # Read file content
        file_content = await file.read()
        actual_size = len(file_content)
        
        # Check size after reading
        if actual_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Generate object key
        object_key = generate_object_key(current_user.id, file.filename)
        
        # Upload to Supabase Storage
        storage_response = supabase_client.upload_file(
            bucket="user-uploads",
            object_key=object_key,
            file_content=file_content,
            file_type=file.content_type
        )
        
        # Create file record in database
        file_record = supabase_client.create_file_object(
            profile_id=current_user.id,
            bucket="user-uploads",
            object_key=object_key,
            mime_type=file.content_type,
            size_bytes=actual_size
        )
        
        # Create file message in the conversation
        file_message_body = {
            "file_id": file_record["id"],
            "filename": file.filename,
            "mime_type": file.content_type,
            "size_bytes": actual_size,
            "public_url": storage_response["public_url"]
        }
        
        message = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="user",
            kind="file",
            body=json.dumps(file_message_body)
        )

        # Get orchestrator and process the file
        orchestrator = get_langgraph_orchestrator()
        
        # Prepare context for the orchestrator
        context = {
            "user_id": current_user.id,
            "conversation_id": str(conversation_id),
            "source_action": "file_uploaded",
        }
        
        # Prepare file info for the orchestrator
        file_info = {
            "file_id": file_record["id"],
            "filename": file.filename,
            "mime_type": file.content_type,
            "size_bytes": actual_size,
            "public_url": storage_response["public_url"],
            "object_key": object_key,
            "bucket": "user-uploads"
        }
        
        # Process the file with the LangGraph orchestrator
        agent_response = await orchestrator.process_user_file(
            conversation_id=str(conversation_id),
            file_info=file_info,
            context=context,
        )
        
        return FileUploadResponse(
            success=True,
            file_id=file_record["id"],
            public_url=storage_response["public_url"],
            message=message,
            response=agent_response.get("response", {}),
            mime_type=file_record["mime_type"],
            size_bytes=file_record["size_bytes"],
            workflow_status=agent_response.get("workflow_status", "unknown"),
            steps_completed=agent_response.get("steps_completed", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )