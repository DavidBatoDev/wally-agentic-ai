# backend/src/models/document.py
"""
Document and file-related schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


class FileObject(BaseModel):
    """Schema for a file object stored in Supabase Storage."""
    id: UUID
    profile_id: UUID
    bucket: str
    object_key: str
    mime_type: str
    size_bytes: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class FileObjectCreate(BaseModel):
    """Schema for creating a new file object."""
    profile_id: UUID
    bucket: str
    object_key: str
    mime_type: str
    size_bytes: int


class DocumentVersion(BaseModel):
    """Schema for a document version."""
    id: UUID
    base_file_id: UUID
    rev: int
    placeholder_json: Dict[str, Any]
    llm_log: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentVersionCreate(BaseModel):
    """Schema for creating a new document version."""
    base_file_id: UUID
    rev: int
    placeholder_json: Dict[str, Any]
    llm_log: Optional[Dict[str, Any]] = None


class Template(BaseModel):
    """Schema for a document template."""
    id: UUID
    doc_type: str
    variation: str
    object_key: str
    placeholder_json: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response after a successful file upload."""
    file_id: UUID
    conversation_id: UUID
    object_key: str
    mime_type: str