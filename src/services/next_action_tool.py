# backend/src/services/next_action_tool.py
"""
Next Action Tool that allows the LLM to recommend follow-up actions as clickable buttons.
"""

from typing import Dict, Any, List, Optional, Awaitable, Callable, Union
from uuid import UUID
import logging
import json

from src.utils.db_client import supabase_client

logger = logging.getLogger(__name__)

async def recommend_next_actions_handler(
    tool_input: Dict[str, Any],
    conversation_id: Optional[UUID] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handler for creating suggested next actions as buttons.
    
    Args:
        tool_input: Dictionary containing the prompt and suggested actions
        conversation_id: The conversation ID
        context: Additional context
        
    Returns:
        Result of creating the buttons message
    """
    try:
        if not conversation_id:
            return {
                "status": "error",
                "error": "Conversation ID is required"
            }
        
        # Extract parameters from the tool input
        prompt = tool_input.get("prompt", "Would you like to explore more?")
        actions = tool_input.get("actions", [])
        
        # Validate actions
        if not actions or not isinstance(actions, list):
            return {
                "status": "error",
                "error": "At least one action must be provided"
            }
        
        # Format buttons for the UI
        buttons = []
        for action in actions:
            action_type = action.get("type", "message")  # Default to message if not specified
            
            button = {
                "text": action.get("text", ""),
                "action": action.get("action", ""),
                "action_type": action_type,
                "parameters": action.get("parameters", {})
            }
            
            # For tool actions, ensure we have the tool name and parameters
            if action_type == "tool":
                if not action.get("tool_name"):
                    continue  # Skip this button if tool_name is missing
                
                button["tool_name"] = action.get("tool_name")
                button["tool_parameters"] = action.get("tool_parameters", {})
            
            buttons.append(button)
        
        if not buttons:
            return {
                "status": "error",
                "error": "No valid buttons could be created"
            }
        
        # Create a buttons message
        message_data = {
            "prompt": prompt,
            "buttons": buttons
        }
        
        # Store the buttons message
        buttons_message = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="assistant",
            kind="buttons",
            body=json.dumps(message_data)
        )
        
        return {
            "status": "success",
            "message_id": buttons_message.get("id"),
            "prompt": prompt,
            "buttons": buttons
        }
        
    except Exception as e:
        logger.error(f"Error in recommend_next_actions_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def register_next_actions_tool(registry):
    """
    Register the next actions tool with the tool registry.
    
    Args:
        registry: The tool registry to register with
    """
    registry.register_tool(
        name="recommend_next_actions",
        description="Recommends follow-up actions to the user as clickable buttons",
        handler=recommend_next_actions_handler,
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt text to display above the buttons"
                },
                "actions": {
                    "type": "array",
                    "description": "List of actions to suggest as buttons",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The button text to display"
                            },
                            "action": {
                                "type": "string",
                                "description": "The action identifier"
                            },
                            "type": {
                                "type": "string",
                                "enum": ["message", "tool"],
                                "description": "The type of action: 'message' for sending a message or 'tool' for calling a tool"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Additional parameters for the action"
                            },
                            "tool_name": {
                                "type": "string",
                                "description": "If type is 'tool', the name of the tool to call"
                            },
                            "tool_parameters": {
                                "type": "object",
                                "description": "If type is 'tool', the parameters to pass to the tool"
                            }
                        },
                        "required": ["text", "action"]
                    }
                }
            },
            "required": ["prompt", "actions"]
        }
    )