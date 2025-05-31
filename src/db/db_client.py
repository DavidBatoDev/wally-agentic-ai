# backend/src/db/db_client.py
"""
Supabase client utilities.
"""

from typing import Any, Dict, List, Optional, Union, cast
from supabase import create_client 
# from postgrest.types import CountMethod
# from postgrest.base_request_builder import APIResponse
import logging

from src.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase client with helper methods for common operations."""
    
    def __init__(self):
        """Initialize the Supabase client."""
        self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    
    """
    Improved Supabase client with fixed get_user_profile method.
    """
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: The user ID
                
        Returns:
            The user profile as a dictionary or None if not found
        """
        try:
            # Using "id" as that's the actual column name in the profiles table
            print(f"Fetching profile for user with id: {user_id}")

            response = (
                self.client.table("profiles")
                .select("*")
                .eq("id", user_id)
                .execute()
                )
            print(f"Response from Supabase: {response}")
            
            # Check if we have any data
            if response.data and len(response.data) > 0:
                print(f"Found profile for user: {user_id}")
                return response.data[0]
            else:
                print(f"No profile found for user: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error in get_user_profile: {str(e)}")
            return None
    
    def create_user_profile(self, user_id: str, full_name: str, avatar_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user profile.
        
        Args:
            user_id: The user ID
            full_name: The user's full name
            avatar_url: Optional URL to the user's avatar
            
        Returns:
            The created profile as a dictionary
        """
        profile_data = {
            "id": user_id, # Assuming "id" is the primary key
            "full_name": full_name,
            "avatar_url": avatar_url
        }
        
        response = self.client.table("profiles").insert(profile_data).execute()
        return response.data[0]
    
    def create_conversation(self, profile_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new conversation.
        
        Args:
            profile_id: The profile ID of the user
            title: Optional title for the conversation
            
        Returns:
            The created conversation as a dictionary
        """
        conversation_data = {
            "profile_id": profile_id,
            "title": title,
            "is_active": True
        }
        
        response = self.client.table("conversations").insert(conversation_data).execute()
        return response.data[0]
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            The conversation as a dictionary
        """
        response = self.client.table("conversations").select("*").eq("id", conversation_id).single().execute()
        return response.data
    
    def get_user_conversations(self, profile_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversations for a user.
        
        Args:
            profile_id: The profile ID of the user
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            
        Returns:
            List of conversations
        """
        response = (
            self.client.table("conversations")
            .select("*")
            .eq("profile_id", profile_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .offset(offset)
            .execute()
        )
        return response.data
    
    def create_message(self, conversation_id: str, sender: str, kind: str, body: str) -> Dict[str, Any]:
        """
        Create a new message in a conversation.
        
        Args:
            conversation_id: The conversation ID
            sender: The sender of the message ("user", "assistant", or "model")
            kind: The kind of message ("text", "file", "action", etc.)
            body: The message body
            
        Returns:
            The created message as a dictionary
        """
        message_data = {
            "conversation_id": conversation_id,
            "sender": sender,
            "kind": kind,
            "body": body
        }
        
        response = self.client.table("messages").insert(message_data).execute()
        return response.data[0]
    
    def get_conversation_messages(self, conversation_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages to return
            offset: Offset for pagination
            
        Returns:
            List of messages
        """
        response = (
            self.client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)  # Oldest first
            .limit(limit)
            .offset(offset)
            .execute()
        )
        return response.data
    
    def create_file_object(
        self,
        profile_id: str,
        bucket: str,
        object_key: str,
        mime_type: str,
        size_bytes: int
    ) -> Dict[str, Any]:
        """
        Create a new file object record.
        
        Args:
            profile_id: The profile ID of the user
            bucket: The storage bucket name
            object_key: The object key in the bucket
            mime_type: The MIME type of the file
            size_bytes: The size of the file in bytes
            
        Returns:
            The created file object as a dictionary
        """
        file_data = {
            "profile_id": profile_id,
            "bucket": bucket,
            "object_key": object_key,
            "mime_type": mime_type,
            "size_bytes": size_bytes
        }
        
        response = self.client.table("file_objects").insert(file_data).execute()
        return response.data[0]
    
    def update_conversation(self, conversation_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a conversation.
        
        Args:
            conversation_id: The conversation ID
            data: The data to update
            
        Returns:
            The updated conversation as a dictionary
        """
        response = (
            self.client.table("conversations")
            .update(data)
            .eq("id", conversation_id)
            .execute()
        )
        
        if not response.data:
            return None
            
        return response.data[0]

    def get_file_object(self, file_id: str) -> Dict[str, Any]:
        """
        Get a file object by ID.
        
        Args:
            file_id: The file object ID
            
        Returns:
            The file object as a dictionary
        """
        response = self.client.table("file_objects").select("*").eq("id", file_id).single().execute()
        return response.data
    
    def create_document_version(
        self,
        base_file_id: str,
        rev: int,
        placeholder_json: Dict[str, Any],
        llm_log: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new document version.
        
        Args:
            base_file_id: The base file ID
            rev: The revision number
            placeholder_json: The placeholder JSON
            llm_log: Optional LLM log data
            
        Returns:
            The created document version as a dictionary
        """
        version_data = {
            "base_file_id": base_file_id,
            "rev": rev,
            "placeholder_json": placeholder_json,
            "llm_log": llm_log
        }
        
        response = self.client.table("doc_versions").insert(version_data).execute()
        return response.data[0]
    
    def get_latest_document_version(self, base_file_id: str) -> Dict[str, Any]:
        """
        Get the latest document version for a base file.
        
        Args:
            base_file_id: The base file ID
            
        Returns:
            The latest document version as a dictionary
        """
        response = (
            self.client.table("doc_versions")
            .select("*")
            .eq("base_file_id", base_file_id)
            .order("rev", desc=True)
            .limit(1)
            .execute()
        )
        
        if not response.data:
            return None
        
        return response.data[0]
    
    # for uploading files to Supabase Storage
    def upload_file(self, bucket: str, object_key: str, file_content: bytes, file_type: str) -> Dict[str, Any]:
        """
        Upload a file to Supabase Storage.
        
        Args:
            bucket: The storage bucket name
            object_key: The object key in the bucket
            file_content: The file content as bytes
            file_type: The MIME type of the file
            
        Returns:
            The response from Supabase Storage
        """
        try:
            # Upload the file to Supabase Storage
            response = self.client.storage.from_(bucket).upload(
                path=object_key,
                file=file_content,
                file_options={"content-type": file_type}
            )
            
            # Get the public URL
            public_url = self.client.storage.from_(bucket).get_public_url(object_key)
            
            # Return a proper dictionary response
            return {
                "key": object_key,
                "public_url": public_url,
                "success": True,
                "path": getattr(response, 'path', object_key) if hasattr(response, 'path') else object_key
            }
        except Exception as e:
            logger.error(f"Error uploading file to Supabase Storage: {str(e)}")
            raise e
        
    def save_conversation_buffer(self, conversation_id: str, buffer_data: List[Dict], buffer_size: int) -> Dict[str, Any]:
        """
        Save or update conversation buffer.
        
        Args:
            conversation_id: The conversation ID
            buffer_data: Serialized message buffer data
            buffer_size: Number of messages in buffer
            
        Returns:
            The saved buffer record as a dictionary
        """
        try:
            upsert_data = {
                "conversation_id": conversation_id,
                "buffer_data": buffer_data,
                "buffer_size": buffer_size
            }
            
            response = (
                self.client.table("conversation_buffer")
                .upsert(upsert_data, on_conflict="conversation_id")
                .execute()
            )
            
            return response.data[0] if response.data else None
            
        except Exception as e:
            logger.error(f"Error saving conversation buffer: {str(e)}")
            raise e

    def get_conversation_buffer(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation buffer by conversation ID.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            The buffer record as a dictionary or None if not found
        """
        try:
            response = (
                self.client.table("conversation_buffer")
                .select("*")
                .eq("conversation_id", conversation_id)
                .single()
                .execute()
            )
            
            return response.data if response.data else None
            
        except Exception as e:
            logger.error(f"Error getting conversation buffer: {str(e)}")
            return None

    def delete_conversation_buffer(self, conversation_id: str) -> bool:
        """
        Delete conversation buffer.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = (
                self.client.table("conversation_buffer")
                .delete()
                .eq("conversation_id", conversation_id)
                .execute()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation buffer: {str(e)}")
            return False

    def get_buffer_stats(self, profile_id: str) -> Dict[str, Any]:
        """
        Get buffer statistics for a user's conversations.
        
        Args:
            profile_id: The profile ID
            
        Returns:
            Dictionary with buffer statistics
        """
        try:
            # Get buffer stats by joining with conversations table
            response = (
                self.client.table("conversation_buffer")
                .select("""
                    buffer_size,
                    conversations!inner(profile_id)
                """)
                .eq("conversations.profile_id", profile_id)
                .execute()
            )
            
            if response.data:
                total_buffers = len(response.data)
                total_messages = sum(item["buffer_size"] for item in response.data)
                avg_buffer_size = total_messages / total_buffers if total_buffers > 0 else 0
                
                return {
                    "total_buffers": total_buffers,
                    "total_messages": total_messages,
                    "average_buffer_size": avg_buffer_size
                }
            
            return {
                "total_buffers": 0,
                "total_messages": 0,
                "average_buffer_size": 0
            }
            
        except Exception as e:
            logger.error(f"Error getting buffer stats: {str(e)}")
            return {
                "total_buffers": 0,
                "total_messages": 0,
                "average_buffer_size": 0
            }

# Singleton instance
supabase_client = SupabaseClient()