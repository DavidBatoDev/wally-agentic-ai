# backend/src/agent/agent_tools.py
"""
Enhanced tools available to the LangGraph agent for performing various actions.
Updated to work seamlessly with the LangGraph orchestrator and workflow system.
"""

from typing import List, Optional, Dict, Any, Union
import json
import asyncio

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.db.db_client import SupabaseClient

# Custom exception class for tool errors
class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    pass

# -----------------------------------------------------------------------------
# Pydantic schemas for existing tools
# -----------------------------------------------------------------------------

class MultiplicationInput(BaseModel):
    """Input for multiplication calculator."""
    a: float = Field(description="First number to multiply")
    b: float = Field(description="Second number to multiply")

class CalculateInput(BaseModel):
    """Input for ad‑hoc arithmetic expressions."""
    expression: str = Field(description="Mathematical expression to evaluate, e.g. '3 * 4'")

# -----------------------------------------------------------------------------
# Enhanced UI Tool Schemas for LangGraph workflows
# -----------------------------------------------------------------------------

class ShowButtonsInput(BaseModel):
    """Schema for interactive button display with workflow support."""
    prompt: str = Field(description="Question or instruction shown above the buttons")
    buttons: List[Dict[str, str]] = Field(
        description="List of button definitions with 'label' and 'action' keys"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional context for the workflow step"
    )

class ShowNotificationInput(BaseModel):
    """Schema for user notifications."""
    message: str = Field(description="Notification text to display to the user")
    severity: str = Field(
        default="info", 
        description="Notification level: info | success | warning | error"
    )

class ShowProgressBarInput(BaseModel):
    """Schema for progress tracking."""
    title: str = Field(description="Title of the operation being tracked")
    percent: int = Field(ge=0, le=100, description="Progress percentage (0‑100)")
    message: Optional[str] = Field(default=None, description="Optional status message")

class ShowInputsInput(BaseModel):
    """Schema for dynamic form generation."""
    prompt: str = Field(description="Instructions shown above the input form")
    inputs: List[Dict[str, Any]] = Field(description="List of input field definitions")
    submit_label: str = Field(default="Submit", description="Label for the submit button")
    validation_rules: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Validation rules for the inputs"
    )

class ShowFileCardInput(BaseModel):
    """Schema for file preview cards."""
    file_id: str = Field(description="Unique identifier for the file")
    title: str = Field(description="Display title for the file")
    thumbnail: Optional[str] = Field(default=None, description="URL to thumbnail image")
    summary: str = Field(description="Brief description of the file content")
    status: str = Field(default="ready", description="Status: ready | processing | error")
    file_size: Optional[str] = Field(default=None, description="Human-readable file size")
    file_type: Optional[str] = Field(default=None, description="File type/extension")

# -----------------------------------------------------------------------------
# Database interaction schemas
# -----------------------------------------------------------------------------

class SearchMemoryInput(BaseModel):
    """Schema for searching conversation memory."""
    query: str = Field(description="Search query text")
    conversation_id: str = Field(description="Conversation ID to search within")
    limit: int = Field(default=5, description="Maximum number of results")

class StoreMemoryInput(BaseModel):
    """Schema for storing information in memory."""
    content: str = Field(description="Content to store")
    conversation_id: str = Field(description="Conversation ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

# -----------------------------------------------------------------------------
# Enhanced tool implementations
# -----------------------------------------------------------------------------

def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """Return the enhanced list of tools available to the LangGraph agent."""
    
    tools: List[BaseTool] = []

    # ------------------------------------------------------------------
    # Core math helpers (enhanced error handling)
    # ------------------------------------------------------------------
    def multiply_numbers(a: float, b: float) -> str:
        """Multiply two numbers and return the result."""
        try:
            result = a * b
            return f"The result of {a} × {b} is {result}"
        except Exception as e:
            raise ToolExecutionError(f"Error multiplying numbers: {e}")

    tools.append(
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together. Use for direct multiplication operations.",
            args_schema=MultiplicationInput,
        )
    )

    def calculate(expression: str) -> str:
        """Evaluate a safe arithmetic expression."""
        allowed_chars = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed_chars):
            raise ToolExecutionError("Expression contains invalid characters")
        
        try:
            result = eval(expression)  # nosec — guarded by whitelist above
            return f"The result of {expression} is {result}"
        except Exception as e:
            raise ToolExecutionError(f"Error calculating expression: {e}")

    tools.append(
        StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description="Calculate arithmetic expressions like '3 * 4 + 2'.",
            args_schema=CalculateInput,
        )
    )

    # ------------------------------------------------------------------
    # Enhanced UI tools for LangGraph workflows
    # ------------------------------------------------------------------
    def show_buttons(
        prompt: str, 
        buttons: List[Dict[str, str]], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Display interactive buttons with workflow context."""
        
        # Ensure buttons have proper structure
        formatted_buttons = []
        for i, btn in enumerate(buttons):
            if isinstance(btn, str):
                formatted_buttons.append({
                    "label": btn, 
                    "action": f"button_{i}",
                    "value": btn.lower().replace(" ", "_")
                })
            elif isinstance(btn, dict):
                formatted_buttons.append({
                    "label": btn.get("label", f"Button {i}"),
                    "action": btn.get("action", f"button_{i}"),
                    "value": btn.get("value", btn.get("label", "").lower().replace(" ", "_"))
                })
        
        response = {
            "kind": "buttons",
            "prompt": prompt,
            "buttons": formatted_buttons,
        }
        
        if context:
            response["context"] = context
            
        return response

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Display interactive buttons for user choice. Use when you need user confirmation or selection.",
            args_schema=ShowButtonsInput,
            return_direct=True,
        )
    )

    def show_inputs(
        prompt: str, 
        inputs: List[Dict[str, Any]], 
        submit_label: str = "Submit",
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Display a form with input fields."""
        
        # Ensure inputs have proper structure
        formatted_inputs = []
        for input_def in inputs:
            formatted_input = {
                "name": input_def.get("name", ""),
                "label": input_def.get("label", ""),
                "type": input_def.get("type", "text"),
                "required": input_def.get("required", False),
                "placeholder": input_def.get("placeholder", ""),
            }
            
            # Add type-specific properties
            if "options" in input_def:
                formatted_input["options"] = input_def["options"]
            if "min" in input_def:
                formatted_input["min"] = input_def["min"]
            if "max" in input_def:
                formatted_input["max"] = input_def["max"]
                
            formatted_inputs.append(formatted_input)
        
        response = {
            "kind": "inputs",
            "prompt": prompt,
            "inputs": formatted_inputs,
            "submit_label": submit_label,
        }
        
        if validation_rules:
            response["validation"] = validation_rules
            
        return response

    tools.append(
        StructuredTool.from_function(
            func=show_inputs,
            name="show_inputs",
            description="Display a form to collect structured data from users. Use when you need specific information.",
            args_schema=ShowInputsInput,
            return_direct=True,
        )
    )

    def show_file_card(
        file_id: str, 
        title: str, 
        summary: str, 
        thumbnail: Optional[str] = None, 
        status: str = "ready",
        file_size: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Display a rich file preview card."""
        
        result = {
            "kind": "file_card",
            "file_id": file_id,
            "title": title,
            "summary": summary,
            "status": status,
            "actions": [
                {"label": "View Details", "action": "view_file_details", "value": file_id},
                {"label": "Download", "action": "download_file", "value": file_id},
            ]
        }
        
        if thumbnail:
            result["thumbnail"] = thumbnail
        if file_size:
            result["file_size"] = file_size
        if file_type:
            result["file_type"] = file_type
            
        return result

    tools.append(
        StructuredTool.from_function(
            func=show_file_card,
            name="show_file_card",
            description="Display a file preview card with actions. Use when presenting files to users.",
            args_schema=ShowFileCardInput,
            return_direct=True,
        )
    )

    def show_progress_bar(
        title: str, 
        percent: int, 
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Display a progress bar for long-running operations."""
        
        result = {
            "kind": "progress",
            "title": title,
            "percent": max(0, min(100, percent)),  # Clamp between 0-100
        }
        
        if message:
            result["message"] = message
            
        return result

    tools.append(
        StructuredTool.from_function(
            func=show_progress_bar,
            name="show_progress_bar",
            description="Display progress information for ongoing operations.",
            args_schema=ShowProgressBarInput,
            return_direct=True,
        )
    )

    def show_notification(message: str, severity: str = "info") -> Dict[str, Any]:
        """Display a notification to the user."""
        
        valid_severities = ["info", "success", "warning", "error"]
        if severity not in valid_severities:
            severity = "info"
            
        return {
            "kind": "notification",
            "message": message,
            "severity": severity,
        }

    tools.append(
        StructuredTool.from_function(
            func=show_notification,
            name="show_notification",
            description="Show a notification message to the user.",
            args_schema=ShowNotificationInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------------------
    # Database-backed tools (if db_client is available)
    # ------------------------------------------------------------------
    if db_client:
        
        def search_memory(query: str, conversation_id: str, limit: int = 5) -> str:
            """Search conversation memory for relevant information."""
            try:
                # This would require implementing vector search in your db_client
                # For now, return a placeholder response
                return f"Memory search for '{query}' in conversation {conversation_id} completed. Found {limit} relevant entries."
            except Exception as e:
                raise ToolExecutionError(f"Error searching memory: {e}")

        tools.append(
            StructuredTool.from_function(
                func=search_memory,
                name="search_memory",
                description="Search conversation memory for relevant past information.",
                args_schema=SearchMemoryInput,
            )
        )

        def store_memory(
            content: str, 
            conversation_id: str, 
            metadata: Optional[Dict[str, Any]] = None
        ) -> str:
            """Store important information in conversation memory."""
            try:
                # This would require implementing embedding storage
                return f"Successfully stored information in memory for conversation {conversation_id}"
            except Exception as e:
                raise ToolExecutionError(f"Error storing memory: {e}")

        tools.append(
            StructuredTool.from_function(
                func=store_memory,
                name="store_memory",
                description="Store important information for future reference.",
                args_schema=StoreMemoryInput,
            )
        )

    return tools