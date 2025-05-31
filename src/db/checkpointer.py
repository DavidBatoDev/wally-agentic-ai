# backend/src/db/checkpointer.py
"""
Enhanced Supabase-based checkpointer for LangGraph message buffers.
Now includes tool calls and outputs for better LLM context.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.db.db_client import supabase_client

logger = logging.getLogger(__name__)


def message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to a dictionary with enhanced tool tracking."""
    base_dict = message.model_dump()
    
    # Add enhanced metadata for different message types
    if isinstance(message, AIMessage):
        # Track tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            base_dict['tool_calls_summary'] = [
                {
                    'name': call.get('name', 'unknown'),
                    'args': call.get('args', {}),
                    'id': call.get('id', 'unknown')
                }
                for call in message.tool_calls
            ]
            base_dict['has_tool_calls'] = True
        else:
            base_dict['has_tool_calls'] = False
    
    elif isinstance(message, ToolMessage):
        # Extract tool result information
        try:
            # Try to parse the content as JSON for structured tool outputs
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            
            # Store high-level tool result summary
            base_dict['tool_result_summary'] = {
                'kind': content.get('kind', 'unknown') if isinstance(content, dict) else 'text',
                'success': True,  # Assume success if we got a result
                'tool_name': getattr(message, 'name', 'unknown'),
                'result_type': type(content).__name__ if not isinstance(content, dict) else content.get('kind', 'dict')
            }
            
            # Store specific tool result details based on type
            if isinstance(content, dict):
                if content.get('kind') == 'buttons':
                    base_dict['tool_result_summary']['buttons_count'] = len(content.get('buttons', []))
                    base_dict['tool_result_summary']['prompt'] = content.get('prompt', '')[:100]  # First 100 chars
                elif content.get('kind') == 'upload_button':
                    base_dict['tool_result_summary']['upload_config'] = {
                        'max_size_mb': content.get('config', {}).get('max_size_mb'),
                        'accepted_types': len(content.get('config', {}).get('accepted_types', []))
                    }
                elif content.get('kind') == 'file_card':
                    base_dict['tool_result_summary']['file_info'] = {
                        'title': content.get('title', '')[:50],
                        'status': content.get('status', 'unknown')
                    }
            
        except (json.JSONDecodeError, AttributeError):
            # Handle non-JSON tool outputs (like calculation results)
            base_dict['tool_result_summary'] = {
                'kind': 'text',
                'success': True,
                'tool_name': getattr(message, 'name', 'unknown'),
                'result_type': 'string',
                'content_preview': str(message.content)[:100] if message.content else ''
            }
    
    # Add timestamp for better tracking
    base_dict['buffer_timestamp'] = datetime.utcnow().isoformat()
    
    return base_dict


def messages_from_dict(messages_data: List[Dict[str, Any]]) -> List[BaseMessage]:
    """Convert a list of message dictionaries back to LangChain messages."""
    messages = []
    for msg_data in messages_data:
        msg_type = msg_data.get("type", "")
        
        # Remove our custom metadata before reconstruction
        clean_msg_data = {k: v for k, v in msg_data.items() 
                         if k not in ['tool_calls_summary', 'has_tool_calls', 'tool_result_summary', 'buffer_timestamp']}
        
        try:
            if msg_type == "human":
                messages.append(HumanMessage.model_validate(clean_msg_data))
            elif msg_type == "ai":
                messages.append(AIMessage.model_validate(clean_msg_data))
            elif msg_type == "system":
                messages.append(SystemMessage.model_validate(clean_msg_data))
            elif msg_type == "tool":
                messages.append(ToolMessage.model_validate(clean_msg_data))
            else:
                # Handle other message types or default to the base class
                logger.warning(f"Unknown message type: {msg_type}")
                continue
        except Exception as e:
            logger.error(f"Error reconstructing message of type {msg_type}: {e}")
            continue
    
    return messages


class SupabaseCheckpointer:
    """
    Enhanced checkpointer that saves and loads LangGraph message buffers to/from Supabase.
    Now includes tool call and output tracking for better LLM context.
    """
    
    def __init__(self):
        """Initialize the checkpointer with Supabase client."""
        self.client = supabase_client
    
    def save_buffer(self, conversation_id: str, messages: List[BaseMessage]) -> bool:
        """
        Save message buffer to Supabase with enhanced tool tracking.
        
        Args:
            conversation_id: The conversation ID
            messages: List of messages to save in buffer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert messages to serializable format with tool tracking
            buffer_data = [message_to_dict(msg) for msg in messages]
            buffer_size = len(messages)
            
            # Generate buffer analytics
            analytics = self._generate_buffer_analytics(buffer_data)
            
            # Prepare data for upsert
            upsert_data = {
                "conversation_id": conversation_id,
                "buffer_data": buffer_data,
                "buffer_size": buffer_size,
                "buffer_analytics": analytics,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Use upsert to insert or update
            response = (
                self.client.client.table("conversation_buffer")
                .upsert(upsert_data, on_conflict="conversation_id")
                .execute()
            )
            
            if response.data:
                logger.info(f"Successfully saved buffer for conversation {conversation_id} with {buffer_size} messages")
                logger.info(f"Buffer analytics: {analytics}")
                return True
            else:
                logger.error(f"Failed to save buffer for conversation {conversation_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving buffer for conversation {conversation_id}: {str(e)}")
            return False
    
    def _generate_buffer_analytics(self, buffer_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analytics about the buffer contents for better context."""
        analytics = {
            "total_messages": len(buffer_data),
            "message_types": {},
            "tool_usage": {
                "messages_with_tools": 0,
                "tool_calls_count": 0,
                "tools_used": {},
                "ui_tools_used": 0,
                "calculation_tools_used": 0
            },
            "conversation_flow": {
                "last_user_message": None,
                "last_assistant_message": None,
                "last_tool_used": None
            }
        }
        
        for msg_data in buffer_data:
            msg_type = msg_data.get("type", "unknown")
            
            # Count message types
            analytics["message_types"][msg_type] = analytics["message_types"].get(msg_type, 0) + 1
            
            # Track tool usage
            if msg_data.get("has_tool_calls"):
                analytics["tool_usage"]["messages_with_tools"] += 1
                tool_calls = msg_data.get("tool_calls_summary", [])
                analytics["tool_usage"]["tool_calls_count"] += len(tool_calls)
                
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    analytics["tool_usage"]["tools_used"][tool_name] = analytics["tool_usage"]["tools_used"].get(tool_name, 0) + 1
            
            # Track tool results
            if msg_data.get("tool_result_summary"):
                tool_summary = msg_data["tool_result_summary"]
                tool_name = tool_summary.get("tool_name", "unknown")
                
                # Update last tool used
                analytics["conversation_flow"]["last_tool_used"] = {
                    "name": tool_name,
                    "kind": tool_summary.get("kind", "unknown"),
                    "success": tool_summary.get("success", False)
                }
                
                # Count UI vs calculation tools
                if tool_summary.get("kind") in ["buttons", "upload_button", "file_card", "inputs"]:
                    analytics["tool_usage"]["ui_tools_used"] += 1
                elif tool_name in ["multiply_numbers", "calculate"]:
                    analytics["tool_usage"]["calculation_tools_used"] += 1
            
            # Track last messages for context
            if msg_type == "human" and msg_data.get("content"):
                analytics["conversation_flow"]["last_user_message"] = msg_data["content"][:100]
            elif msg_type == "ai" and msg_data.get("content"):
                analytics["conversation_flow"]["last_assistant_message"] = msg_data["content"][:100]
        
        return analytics
    
    def load_buffer(self, conversation_id: str) -> List[BaseMessage]:
        """
        Load message buffer from Supabase.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of messages from buffer, empty list if none found
        """
        try:
            response = (
                self.client.client.table("conversation_buffer")
                .select("buffer_data, buffer_size, buffer_analytics")
                .eq("conversation_id", conversation_id)
                .single()
                .execute()
            )
            
            if response.data and response.data.get("buffer_data"):
                buffer_data = response.data["buffer_data"]
                buffer_analytics = response.data.get("buffer_analytics", {})
                messages = messages_from_dict(buffer_data)
                
                logger.info(f"Loaded buffer for conversation {conversation_id} with {len(messages)} messages")
                
                # Log helpful analytics for debugging
                if buffer_analytics:
                    tool_usage = buffer_analytics.get("tool_usage", {})
                    logger.info(f"Buffer contains {tool_usage.get('tool_calls_count', 0)} tool calls across {tool_usage.get('messages_with_tools', 0)} messages")
                    if tool_usage.get("tools_used"):
                        logger.info(f"Tools used: {tool_usage['tools_used']}")
                
                return messages
            else:
                logger.info(f"No buffer found for conversation {conversation_id}")
                return []
                
        except Exception as e:
            # Log the error but don't raise - return empty buffer instead
            logger.error(f"Error loading buffer for conversation {conversation_id}: {str(e)}")
            return []
    
    def get_buffer_context_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a high-level summary of the conversation context including tool usage.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dictionary with context summary or None if not found
        """
        try:
            response = (
                self.client.client.table("conversation_buffer")
                .select("buffer_analytics, buffer_size, updated_at")
                .eq("conversation_id", conversation_id)
                .single()
                .execute()
            )
            
            if response.data:
                analytics = response.data.get("buffer_analytics", {})
                
                return {
                    "buffer_size": response.data["buffer_size"],
                    "updated_at": response.data["updated_at"],
                    "message_distribution": analytics.get("message_types", {}),
                    "tool_usage_summary": analytics.get("tool_usage", {}),
                    "recent_context": analytics.get("conversation_flow", {}),
                    "context_quality": self._assess_context_quality(analytics)
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting buffer context summary for conversation {conversation_id}: {str(e)}")
            return None
    
    def _assess_context_quality(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and richness of the conversation context."""
        tool_usage = analytics.get("tool_usage", {})
        flow = analytics.get("conversation_flow", {})
        
        quality_score = 0
        quality_factors = []
        
        # Factor 1: Tool usage diversity
        tools_used_count = len(tool_usage.get("tools_used", {}))
        if tools_used_count > 0:
            quality_score += min(tools_used_count * 10, 30)
            quality_factors.append(f"{tools_used_count} different tools used")
        
        # Factor 2: Conversation completeness
        if flow.get("last_user_message") and flow.get("last_assistant_message"):
            quality_score += 20
            quality_factors.append("Complete conversation flow")
        
        # Factor 3: Recent tool usage
        if flow.get("last_tool_used"):
            quality_score += 15
            quality_factors.append("Recent tool interaction")
        
        # Factor 4: Message volume
        total_messages = analytics.get("total_messages", 0)
        if total_messages > 10:
            quality_score += 10
            quality_factors.append("Rich conversation history")
        
        return {
            "score": min(quality_score, 100),
            "factors": quality_factors,
            "recommendation": self._get_context_recommendation(quality_score)
        }
    
    def _get_context_recommendation(self, quality_score: int) -> str:
        """Get a recommendation based on context quality."""
        if quality_score >= 70:
            return "Excellent context - LLM should have rich understanding"
        elif quality_score >= 50:
            return "Good context - LLM should perform well"
        elif quality_score >= 30:
            return "Moderate context - Consider asking for clarification"
        else:
            return "Limited context - May need more information"
    
    def clear_buffer(self, conversation_id: str) -> bool:
        """
        Clear the message buffer for a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = (
                self.client.client.table("conversation_buffer")
                .delete()
                .eq("conversation_id", conversation_id)
                .execute()
            )
            
            logger.info(f"Cleared buffer for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing buffer for conversation {conversation_id}: {str(e)}")
            return False
    
    def get_buffer_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get buffer metadata (size, timestamps, analytics) without loading full buffer.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dictionary with buffer info or None if not found
        """
        try:
            response = (
                self.client.client.table("conversation_buffer")
                .select("buffer_size, buffer_analytics, created_at, updated_at")
                .eq("conversation_id", conversation_id)
                .single()
                .execute()
            )
            
            if response.data:
                return {
                    "buffer_size": response.data["buffer_size"],
                    "buffer_analytics": response.data.get("buffer_analytics", {}),
                    "created_at": response.data["created_at"],
                    "updated_at": response.data["updated_at"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting buffer info for conversation {conversation_id}: {str(e)}")
            return None
    
    def add_message_to_buffer(self, conversation_id: str, message: BaseMessage) -> bool:
        """
        Add a single message to existing buffer.
        
        Args:
            conversation_id: The conversation ID
            message: The message to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing buffer
            existing_messages = self.load_buffer(conversation_id)
            
            # Add new message
            existing_messages.append(message)
            
            # Save updated buffer
            return self.save_buffer(conversation_id, existing_messages)
            
        except Exception as e:
            logger.error(f"Error adding message to buffer for conversation {conversation_id}: {str(e)}")
            return False
    
    def get_recent_messages(self, conversation_id: str, limit: int = 10) -> List[BaseMessage]:
        """
        Get the most recent messages from buffer.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of recent messages to return
            
        Returns:
            List of recent messages
        """
        try:
            messages = self.load_buffer(conversation_id)
            
            # Return the last 'limit' messages
            return messages[-limit:] if messages else []
            
        except Exception as e:
            logger.error(f"Error getting recent messages for conversation {conversation_id}: {str(e)}")
            return []
    
    def trim_buffer(self, conversation_id: str, max_size: int) -> bool:
        """
        Trim buffer to maximum size, keeping most recent messages.
        
        Args:
            conversation_id: The conversation ID
            max_size: Maximum number of messages to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            messages = self.load_buffer(conversation_id)
            
            if len(messages) > max_size:
                # Keep only the most recent messages
                trimmed_messages = messages[-max_size:]
                return self.save_buffer(conversation_id, trimmed_messages)
            
            return True  # No trimming needed
            
        except Exception as e:
            logger.error(f"Error trimming buffer for conversation {conversation_id}: {str(e)}")
            return False


# Singleton instance
supabase_checkpointer = SupabaseCheckpointer()