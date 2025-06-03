# backend/src/agent/helpers.py

import json
import logging
from typing import Any, Dict, Optional

from src.db.db_client import SupabaseClient

log = logging.getLogger(__name__)


def should_create_message(
    db_client: SupabaseClient, 
    conversation_id: str, 
    final_resp: Dict[str, Any]
) -> bool:
    """
    Check if we should create a new message by comparing with the last assistant message.
    Returns True if we should create the message, False if it's a duplicate.
    
    Args:
        db_client: The Supabase database client
        conversation_id: The conversation ID to check
        final_resp: The response dict that would be created as a message
        
    Returns:
        bool: True if message should be created, False if it's a duplicate
    """
    try:
        # Get the last assistant message from the database
        result = (
            db_client.client.table("messages")
            .select("body, kind")
            .eq("conversation_id", conversation_id)
            .eq("sender", "assistant")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if not result.data:
            # No previous assistant messages, safe to create
            return True
        
        last_message = result.data[0]
        last_body = json.loads(last_message["body"]) if last_message["body"] else {}
        last_kind = last_message["kind"]
        
        # Compare the content
        current_kind = final_resp.get("kind")
        
        # If kinds are different, create message
        if last_kind != current_kind:
            return True
        
        # For text messages, compare the text content
        if current_kind == "text":
            last_text = last_body.get("text", "").strip()
            current_text = final_resp.get("text", "").strip()
            return last_text != current_text
        
        # For other message types (buttons, file_card, etc.), do a full comparison
        # excluding any timestamp or dynamic fields that might differ
        normalized_last = _normalize_message_for_comparison(last_body)
        normalized_current = _normalize_message_for_comparison(final_resp)
        
        return normalized_last != normalized_current
        
    except Exception as e:
        log.warning(f"Error checking for duplicate message in conversation {conversation_id}: {e}")
        # If we can't check, err on the side of creating the message
        return True


def _normalize_message_for_comparison(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove dynamic fields that shouldn't affect duplicate detection.
    
    Args:
        data: The message data dictionary
        
    Returns:
        dict: Normalized data with dynamic fields removed
    """
    if not isinstance(data, dict):
        return data
    
    normalized = data.copy()
    
    # Remove common dynamic fields that change between identical messages
    dynamic_fields = [
        "timestamp", 
        "id", 
        "created_at", 
        "updated_at",
        "message_id",
        "conversation_id",  # This will be the same anyway for comparison
        "sender"  # This will be "assistant" for both
    ]
    
    for field in dynamic_fields:
        normalized.pop(field, None)
    
    return normalized


def create_message_if_not_duplicate(
    db_client: SupabaseClient,
    conversation_id: str,
    sender: str,
    kind: str,
    final_resp: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Create a message only if it's not a duplicate of the last assistant message.
    
    Args:
        db_client: The Supabase database client
        conversation_id: The conversation ID
        sender: The message sender (usually "assistant")
        kind: The message kind
        final_resp: The response dict to create as a message
        
    Returns:
        dict or None: The created message dict, or None if it was a duplicate
    """
    if sender == "assistant" and not should_create_message(db_client, conversation_id, final_resp):
        log.info(f"Skipping duplicate message creation for conversation {conversation_id}")
        return None
    
    # Create the message
    return db_client.create_message(
        conversation_id=conversation_id,
        sender=sender,
        kind=kind,
        body=json.dumps(final_resp),
    )