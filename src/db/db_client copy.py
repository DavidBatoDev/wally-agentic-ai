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
    
    def store_memory(
        self,
        conversation_id: str,
        content: str,
        embedding: List[float],
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a memory entry in the RAG store.
        
        Args:
            conversation_id: The conversation ID
            content: The text content
            embedding: The vector embedding
            meta: Optional metadata
            
        Returns:
            The created memory entry as a dictionary
        """
        memory_data = {
            "conversation_id": conversation_id,
            "content": content,
            "embedding": embedding,
            "meta": meta or {}
        }

        if not self._should_embed(meta.get("sender"), meta.get("type", "text")):
            return  # skip embedding entirely
        
        response = self.client.table("rag_memory").insert(memory_data).execute()
        return response.data[0]
    
    # Simplified version of the should_embed function, in the future call llm for gatekeeping
    def _should_embed(self, sender: str, kind: str) -> bool:
        # Simulation of a decision-making process for embedding
        return sender == "user" and kind == "text"
    
    def search_memory(
        self,
        conversation_id: str,
        embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            conversation_id: The conversation ID
            embedding: The query embedding vector
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar memory entries
        """
        # Using RPC for vector search since PostgREST doesn't support it directly
        response = self.client.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_threshold": similarity_threshold,
                "match_count": limit,
                "p_conversation_id": conversation_id
            }
        ).execute()
        
        return response.data

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
            
            return {
                "key": object_key,
                "public_url": public_url,
                **response
            }
        except Exception as e:
            logger.error(f"Error uploading file to Supabase Storage: {str(e)}")
            raise e
        
        # For our agentic AI simple memory system
    def load_conversation_memory(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation memory from database.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Memory data or None if not found
        """
        try:
            response = (
                self.client.table("conversation_memory")
                .select("*")
                .eq("conversation_id", conversation_id)
                .single()
                .execute()
            )
            
            if response.data:
                return response.data
            return None
            
        except Exception as e:
            logger.info(f"No existing memory for conversation {conversation_id}: {e}")
            return None

    def save_conversation_memory(
        self, 
        conversation_id: str, 
        summary: str = "", 
        important_facts: List[str] = None, 
        recent_tools_used: List[Dict[str, Any]] = None,
        user_preferences: Dict[str, Any] = None,
        message_count: int = 0
    ) -> Dict[str, Any]:
        """
        Save or update conversation memory.
        
        Args:
            conversation_id: The conversation ID
            summary: Conversation summary
            important_facts: List of important facts to remember
            recent_tools_used: List of recently used tools
            user_preferences: User preferences learned
            message_count: Current message count
            
        Returns:
            The saved memory data
        """
        try:
            # Prepare data
            memory_data = {
                "conversation_id": conversation_id,
                "summary": summary or "",
                "important_facts": important_facts or [],
                "recent_tools_used": recent_tools_used or [],
                "user_preferences": user_preferences or {},
                "message_count": message_count,
                "last_summary_at": "now()" if summary else None
            }
            
            # Try to update existing record first
            existing = self.load_conversation_memory(conversation_id)
            
            if existing:
                # Update existing
                response = (
                    self.client.table("conversation_memory")
                    .update(memory_data)
                    .eq("conversation_id", conversation_id)
                    .execute()
                )
            else:
                # Insert new
                response = (
                    self.client.table("conversation_memory")
                    .insert(memory_data)
                    .execute()
                )
            
            return response.data[0] if response.data else None
            
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}")
            raise e

    def get_conversation_messages_with_limit(
        self, 
        conversation_id: str, 
        limit: int = 30,
        offset: int = 0,
        exclude_ui_tools: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages with smart filtering for context management.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of messages
            offset: Offset for pagination
            exclude_ui_tools: Whether to exclude UI tool responses
            
        Returns:
            List of messages suitable for LLM context
        """
        try:
            response = (
                self.client.table("messages")
                .select("*")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=False)  # Oldest first
                .limit(limit)
                .offset(offset)
                .execute()
            )
            
            messages = response.data or []
            
            if exclude_ui_tools:
                # Filter out UI tool responses that don't add context
                filtered_messages = []
                ui_tool_kinds = {"buttons", "inputs", "file_card", "progress", "notification"}
                
                for msg in messages:
                    try:
                        # Skip UI tool responses
                        if msg.get('sender') == 'assistant' and msg.get('kind') in ui_tool_kinds:
                            continue
                        
                        # Keep text messages and meaningful tool results
                        filtered_messages.append(msg)
                        
                    except Exception:
                        # Keep message if we can't parse it
                        filtered_messages.append(msg)
                
                return filtered_messages
            
            return messages
            
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
            return []

    def store_simple_memory(
        self,
        conversation_id: str,
        content: str,
        content_type: str = "message",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store content in simple memory for future retrieval.
        
        Args:
            conversation_id: The conversation ID
            content: The content to store
            content_type: Type of content (message, tool_result, summary)
            metadata: Additional metadata
            
        Returns:
            The stored memory entry
        """
        try:
            memory_data = {
                "conversation_id": conversation_id,
                "content": content,
                "content_type": content_type,
                "metadata": metadata or {}
            }
            
            response = (
                self.client.table("simple_memory_store")
                .insert(memory_data)
                .execute()
            )
            
            return response.data[0] if response.data else None
            
        except Exception as e:
            logger.error(f"Error storing simple memory: {e}")
            return None

    def search_simple_memory(
        self,
        conversation_id: str,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search simple memory using text search.
        
        Args:
            conversation_id: The conversation ID
            query: Search query
            content_type: Optional content type filter
            limit: Maximum results
            
        Returns:
            List of matching memory entries
        """
        try:
            # Build query
            query_builder = (
                self.client.table("simple_memory_store")
                .select("*")
                .eq("conversation_id", conversation_id)
                .textSearch("content", query)
            )
            
            if content_type:
                query_builder = query_builder.eq("content_type", content_type)
            
            response = (
                query_builder
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error searching simple memory: {e}")
            return []

    def get_message_count(self, conversation_id: str) -> int:
        """
        Get total message count for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Total message count
        """
        try:
            response = (
                self.client.table("messages")
                .select("id", count="exact")
                .eq("conversation_id", conversation_id)
                .execute()
            )
            
            return response.count or 0
            
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0

    def cleanup_old_simple_memory(self, conversation_id: str, keep_recent: int = 100) -> int:
        """
        Clean up old simple memory entries to prevent bloat.
        
        Args:
            conversation_id: The conversation ID
            keep_recent: Number of recent entries to keep
            
        Returns:
            Number of entries deleted
        """
        try:
            # Get IDs of entries to keep
            keep_response = (
                self.client.table("simple_memory_store")
                .select("id")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=True)
                .limit(keep_recent)
                .execute()
            )
            
            if not keep_response.data:
                return 0
            
            keep_ids = [entry["id"] for entry in keep_response.data]
            
            # Delete old entries
            delete_response = (
                self.client.table("simple_memory_store")
                .delete()
                .eq("conversation_id", conversation_id)
                .not_.in_("id", keep_ids)
                .execute()
            )
            
            return len(delete_response.data) if delete_response.data else 0
            
        except Exception as e:
            logger.error(f"Error cleaning up memory: {e}")
            return 0

# Singleton instance
supabase_client = SupabaseClient()