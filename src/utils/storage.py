# backend/src/utils/storage.py
"""
Storage utilities for file handling.
"""

import os
import uuid
from typing import BinaryIO, Dict, Any, Optional
from fastapi import UploadFile

from utils.db_client import supabase_client
from src.config import get_settings

settings = get_settings()


class StorageManager:
    """Manager for file storage operations."""
    def __init__(self):
        """Initialize the storage manager."""
        self.storage_client = supabase_client.client.storage
        self.bucket_name = settings.STORAGE_BUCKET_NAME
        
        # The bucket should be created via the Supabase dashboard
        # This code assumes the bucket already exists
        pass
    
    def _generate_object_key(self, conversation_id: str, filename: str) -> str:
        """
        Generate a unique object key for a file.
        
        Args:
            conversation_id: The conversation ID
            filename: The original filename
            
        Returns:
            A unique object key
        """
        file_uuid = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1].lower()
        return f"docs/{conversation_id}/{file_uuid}{ext}"
    
    async def upload_file(
        self,
        file: UploadFile,
        profile_id: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """
        Upload a file to storage and create a file_object record.
        
        Args:
            file: The uploaded file
            profile_id: The profile ID of the user
            conversation_id: The conversation ID
            
        Returns:
            The created file object
        """
        # Generate a unique object key
        object_key = self._generate_object_key(conversation_id, file.filename)
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Upload to storage
        self.storage_client.from_(self.bucket_name).upload(
            path=object_key,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        # Create file_object record
        file_object = supabase_client.create_file_object(
            profile_id=profile_id,
            bucket=self.bucket_name,
            object_key=object_key,
            mime_type=file.content_type,
            size_bytes=file_size
        )
        
        return file_object
    
    def get_download_url(self, object_key: str, expires_in: int = 3600) -> str:
        """
        Get a signed download URL for a file.
        
        Args:
            object_key: The object key in the bucket
            expires_in: Expiration time in seconds
            
        Returns:
            Signed download URL
        """
        return self.storage_client.from_(self.bucket_name).create_signed_url(
            path=object_key,
            expires_in=expires_in
        )["signedURL"]
    
    def download_file(self, object_key: str) -> bytes:
        """
        Download a file from storage.
        
        Args:
            object_key: The object key in the bucket
            
        Returns:
            File content as bytes
        """
        response = self.storage_client.from_(self.bucket_name).download(object_key)
        return response


# Singleton instance
storage_manager = StorageManager()

# Export the upload_file function as upload_file_to_storage for backward compatibility
upload_file_to_storage = storage_manager.upload_file