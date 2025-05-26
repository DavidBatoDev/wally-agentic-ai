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

    def show_inputs(
        prompt: str, 
        inputs: List[Dict[str, Any]], 
        submit_label: str = "Submit",
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> str:
        """Display a form with input fields."""
        
        print(f"Showing input form with prompt: {prompt}")
        
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
            
        return json.dumps(response)

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
    ) -> str:
        """Display a rich file preview card."""
        
        print(f"Showing file card for: {title}")
        
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
            
        return json.dumps(result)

    tools.append(
        StructuredTool.from_function(
            func=show_file_card,
            name="show_file_card",
            description="Display a file preview card with actions. Use when presenting files to users.",
            args_schema=ShowFileCardInput,
            return_direct=True,
        )
    )

    return tools