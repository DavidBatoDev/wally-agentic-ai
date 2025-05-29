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

class ShowUploadButtonInput(BaseModel):
    """Schema for file upload button display."""
    prompt: str = Field(description="Instructions or message shown above the upload button")
    accepted_types: Optional[List[str]] = Field(
        default=None, 
        description="List of accepted file types (e.g., ['image/*', 'application/pdf'])"
    )
    max_size_mb: Optional[int] = Field(
        default=10, 
        description="Maximum file size in MB"
    )
    multiple: bool = Field(
        default=False, 
        description="Whether to allow multiple file selection"
    )
    upload_endpoint: Optional[str] = Field(
        default="/api/uploads/file", 
        description="API endpoint for file upload"
    ),
    conversation_id: str = Field(
        default=None, 
        description="Conversation ID to associate with the uploaded files"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional context for the upload workflow"
    )

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
            print(f"Multiplying {a} × {b} = {result}")
            return f"The result of {a} × {b} is {result}"
        except Exception as e:
            print(f"Error multiplying numbers: {e}")
            raise ToolExecutionError(f"Error multiplying numbers: {e}")

    tools.append(
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together. Use this when someone asks for multiplication like '3 * 3' or 'what is 5 times 7'.",
            args_schema=MultiplicationInput,
        )
    )

    def calculate(expression: str) -> str:
        """Evaluate a safe arithmetic expression."""
        allowed_chars = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed_chars):
            raise ToolExecutionError("Expression contains invalid characters")
        
        try:
            print(f"Calculating expression: {expression}")
            result = eval(expression)  # nosec — guarded by whitelist above
            print(f"Result: {result}")
            return f"The result of {expression} is {result}"
        except Exception as e:
            print(f"Error calculating expression: {e}")
            raise ToolExecutionError(f"Error calculating expression: {e}")

    tools.append(
        StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description="Calculate arithmetic expressions like '3 * 4 + 2' or '(10 - 5) * 2'. Use this for complex math expressions.",
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
    ) -> str:
        """Display interactive buttons with workflow context."""
        
        print(f"Showing buttons with prompt: {prompt}")
        
        # Ensure buttons have proper structure for consistent action handling
        formatted_buttons = []
        for i, btn in enumerate(buttons):
            if isinstance(btn, str):
                # Simple string button
                button_value = btn.lower().replace(" ", "_")
                formatted_buttons.append({
                    "label": btn,
                    "action": f"button_{button_value}",
                    "value": button_value,
                    "text": btn  # The text that will be sent as the message
                })
            elif isinstance(btn, dict):
                # Dictionary button with more control
                label = btn.get("label", f"Button {i}")
                value = btn.get("value", label.lower().replace(" ", "_"))
                
                formatted_buttons.append({
                    "label": label,
                    "action": btn.get("action", f"button_{value}"),
                    "value": value,
                    "text": btn.get("text", label)  # What gets sent as the message
                })
        
        response = {
            "kind": "buttons",
            "prompt": prompt,
            "buttons": formatted_buttons,
            "instructions": "Click a button to continue the conversation"
        }
        
        if context:
            response["context"] = context
            
        print(f"Created {len(formatted_buttons)} buttons")
        return json.dumps(response)

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Display interactive buttons for user choice. Use when you need user confirmation or selection.",
            args_schema=ShowButtonsInput,
            return_direct=True,
        )
    )

    def show_upload_button(
        prompt: str,
        accepted_types: Optional[List[str]] = None,
        max_size_mb: int = 10,
        multiple: bool = False,
        upload_endpoint: str = "/api/uploads/file",
        conversation_id: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Display a file upload button with specified constraints and upload endpoint."""
        
        print(f"Showing upload button with prompt: {prompt}")
        
        # Set default accepted types based on your upload router
        if accepted_types is None:
            accepted_types = [
                "image/jpeg", "image/jpg", "image/png", "image/gif", 
                "image/webp", "image/bmp", "image/tiff", "application/pdf"
            ]
        
        # Create the upload button response
        response = {
            "kind": "upload_button",
            "prompt": prompt,
            "config": {
                "accepted_types": accepted_types,
                "max_size_mb": max_size_mb,
                "multiple": multiple,
                "upload_endpoint": upload_endpoint,
                "method": "POST",
                "headers": {
                    "Content-Type": "multipart/form-data"
                },
                "conversation_id": conversation_id if conversation_id else None
            },
            "ui": {
                "button_text": "Choose Files" if multiple else "Choose File",
                "drag_drop_text": f"Drag and drop {'files' if multiple else 'a file'} here or click to browse",
                "max_size_text": f"Maximum file size: {max_size_mb}MB",
                "accepted_types_text": "Accepted formats: " + ", ".join([
                    t.split("/")[-1].upper() if "/" in t else t.upper() 
                    for t in accepted_types
                ])
            },
        }
        
        if context:
            response["context"] = context
            
        print(f"Upload button configured: {len(accepted_types)} types, {max_size_mb}MB max")
        return json.dumps(response)

    tools.append(
        StructuredTool.from_function(
            func=show_upload_button,
            name="show_upload_button",
            description="Display a file upload button. Use when the user wants to upload files, images, or documents. Integrates with your upload API.",
            args_schema=ShowUploadButtonInput,
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
    ) -> str:
        """Display a file preview card."""
        
        print(f"Showing file card for: {title}")
        
        response = {
            "kind": "file_card",
            "file_id": file_id,
            "title": title,
            "summary": summary,
            "status": status,
        }
        
        if thumbnail:
            response["thumbnail"] = thumbnail
        if file_size:
            response["file_size"] = file_size
        if file_type:
            response["file_type"] = file_type
            
        return json.dumps(response)

    tools.append(
        StructuredTool.from_function(
            func=show_file_card,
            name="show_file_card",
            description="Display a file preview card with metadata. Use after file upload or when showing file information.",
            args_schema=ShowFileCardInput,
            return_direct=True,
        )
    )

    return tools