# backend/src/agent/utils.py
"""
Utility functions for LangGraph orchestrator operations.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from enum import Enum

# Fix the import - use relative import or handle the import error gracefully
try:
    from .agent_state import WorkflowStatus
except ImportError:
    try:
        from agent_state import WorkflowStatus
    except ImportError:
        # Define minimal enums if import fails
        from enum import Enum
        
    class WorkflowStatus(str, Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        WAITING_CONFIRMATION = "waiting_confirmation"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        

class WorkflowStep:
    def __init__(self, status=WorkflowStatus.PENDING):
        self.status = status


def format_workflow_response(
    response: Dict[str, Any], 
    include_debug: bool = False
) -> Dict[str, Any]:
    """
    Format a workflow response for API consumption.
    
    Args:
        response: Raw response from LangGraph orchestrator
        include_debug: Whether to include debug information
        
    Returns:
        Formatted response dictionary
    """
    formatted = {
        "success": True,
        "message": response.get("message"),
        "response": response.get("response", {"kind": "text", "text": "Operation completed."}),
        "workflow_status": response.get("workflow_status", "completed"),
        "steps_completed": response.get("steps_completed", 0),
        "user_confirmation_pending": response.get("user_confirmation_pending", False)
    }
    
    if include_debug:
        formatted["debug"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "raw_response": response
        }
    
    return formatted


def create_error_response(
    error: Exception, 
    conversation_id: str,
    fallback_message: str = "I'm sorry, I encountered an error while processing your request."
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: The exception that occurred
        conversation_id: The conversation ID
        fallback_message: Fallback message to show users
        
    Returns:
        Formatted error response
    """
    return {
        "success": False,
        "error": str(error),
        "conversation_id": conversation_id,
        "response": {
            "kind": "text",
            "text": fallback_message
        },
        "workflow_status": "failed",
        "steps_completed": 0,
        "user_confirmation_pending": False
    }


def validate_tool_response(response: Any) -> bool:
    """
    Validate that a tool response has the expected structure.
    
    Args:
        response: Response from a tool
        
    Returns:
        True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    required_fields = ["kind"]
    return all(field in response for field in required_fields)


def extract_user_action_from_message(message_body: str) -> Optional[Dict[str, Any]]:
    """
    Extract user action data from a message body.
    
    Args:
        message_body: JSON string containing action data
        
    Returns:
        Parsed action data or None if invalid
    """
    try:
        data = json.loads(message_body)
        if "action" in data:
            return {
                "action": data["action"],
                "values": data.get("values", {}),
                "source_message_id": data.get("source_message_id")
            }
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def create_workflow_summary(steps: List[WorkflowStep]) -> Dict[str, Any]:
    """
    Create a summary of workflow execution.
    
    Args:
        steps: List of workflow steps
        
    Returns:
        Workflow summary
    """
    total_steps = len(steps)
    completed_steps = len([s for s in steps if s.status == WorkflowStatus.COMPLETED])
    failed_steps = len([s for s in steps if s.status == WorkflowStatus.FAILED])
    pending_steps = len([s for s in steps if s.status == WorkflowStatus.PENDING])
    
    return {
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "pending_steps": pending_steps,
        "completion_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
        "has_failures": failed_steps > 0,
        "all_completed": completed_steps == total_steps and total_steps > 0
    }


def merge_context(
    base_context: Dict[str, Any], 
    additional_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Safely merge context dictionaries.
    
    Args:
        base_context: Base context dictionary
        additional_context: Additional context to merge
        
    Returns:
        Merged context dictionary
    """
    merged = base_context.copy()
    
    for key, value in additional_context.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_context(merged[key], value)
        else:
            merged[key] = value
    
    return merged


class LangGraphResponseHandler:
    """
    Helper class for handling LangGraph responses consistently.
    """
    
    @staticmethod
    def handle_success(
        response: Dict[str, Any],
        message_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle successful responses."""
        result = format_workflow_response(response)
        if message_data:
            result.update(message_data)
        return result
    
    @staticmethod
    def handle_error(
        error: Exception,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle error responses."""
        result = create_error_response(error, conversation_id)
        if context:
            result["context"] = context
        return result
    
    @staticmethod
    def handle_confirmation_required(
        response: Dict[str, Any],
        confirmation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle responses requiring user confirmation."""
        result = format_workflow_response(response)
        result["user_confirmation_pending"] = True
        result["confirmation_data"] = confirmation_data
        return result


# Language code to full name mapping
LANGUAGE_MAP = {
    # Most common languages
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'el': 'Greek',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'tl': 'Filipino',
    'uk': 'Ukrainian',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mt': 'Maltese',
    'ga': 'Irish',
    'cy': 'Welsh',
    'is': 'Icelandic',
    'fa': 'Persian',
    'ur': 'Urdu',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'si': 'Sinhalese',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'ka': 'Georgian',
    'am': 'Amharic',
    'sw': 'Swahili',
    'zu': 'Zulu',
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'mk': 'Macedonian',
    'az': 'Azerbaijani',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'uz': 'Uzbek',
    'tg': 'Tajik',
    'mn': 'Mongolian',
    'hy': 'Armenian',
    
    # Regional variants
    'en-us': 'English (US)',
    'en-gb': 'English (UK)',
    'en-ca': 'English (Canada)',
    'en-au': 'English (Australia)',
    'es-es': 'Spanish (Spain)',
    'es-mx': 'Spanish (Mexico)',
    'es-ar': 'Spanish (Argentina)',
    'fr-fr': 'French (France)',
    'fr-ca': 'French (Canada)',
    'pt-br': 'Portuguese (Brazil)',
    'pt-pt': 'Portuguese (Portugal)',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'de-de': 'German (Germany)',
    'de-at': 'German (Austria)',
    'de-ch': 'German (Switzerland)',
}


def normalize_language(lang_code):
    """
    Normalizes a language code to its full language name.
    
    Args:
        lang_code (str): The language code (e.g., 'en', 'es', 'fr')
    
    Returns:
        str: The full language name, or the original code if not found
    
    Examples:
        >>> normalize_language('en')
        'English'
        >>> normalize_language('es-mx')
        'Spanish (Mexico)'
        >>> normalize_language('xyz')
        'xyz'
    """
    if not isinstance(lang_code, str):
        return lang_code
    
    # Convert to lowercase for case-insensitive matching
    normalized_code = lang_code.lower()
    
    # Try exact match first
    if normalized_code in LANGUAGE_MAP:
        return LANGUAGE_MAP[normalized_code]
    
    # If no exact match and it's a regional variant (contains hyphen),
    # try the base language code
    if '-' in normalized_code:
        base_code = normalized_code.split('-')[0]
        if base_code in LANGUAGE_MAP:
            return LANGUAGE_MAP[base_code]
    
    # Return original code if not found
    return lang_code


def get_supported_language_codes():
    """
    Get all supported language codes.
    
    Returns:
        list: List of supported language codes
    """
    return list(LANGUAGE_MAP.keys())


def get_supported_language_names():
    """
    Get all supported language names.
    
    Returns:
        list: List of supported language names
    """
    return list(LANGUAGE_MAP.values())


def is_language_supported(lang_code):
    """
    Check if a language code is supported.
    
    Args:
        lang_code (str): The language code to check
        
    Returns:
        bool: True if supported, False otherwise
    """
    if not isinstance(lang_code, str):
        return False
    
    normalized_code = lang_code.lower()
    if normalized_code in LANGUAGE_MAP:
        return True
    
    # Check base code for regional variants
    if '-' in normalized_code:
        base_code = normalized_code.split('-')[0]
        return base_code in LANGUAGE_MAP
    
    return False
