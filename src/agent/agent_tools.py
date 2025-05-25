"""
Tools available to the agent for performing various actions.
Added several mock UI‑oriented tools that just return a fixed payload so you can prototype agent responses without
wiring up a real front‑end. Nothing here triggers external side‑effects; everything is fully deterministic and
sandbox‑safe.
"""

from typing import List, Optional, Dict, Any

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.db.db_client import SupabaseClient

# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class MultiplicationInput(BaseModel):
    """Input for multiplication calculator."""

    a: float = Field(description="First number to multiply")
    b: float = Field(description="Second number to multiply")


class CalculateInput(BaseModel):
    """Input for ad‑hoc arithmetic expressions."""

    expression: str = Field(description="Mathematical expression to evaluate, e.g. '3 * 4'")


# ===  UI MOCK TOOL SCHEMAS ====================================================
class ShowButtonsInput(BaseModel):
    """Schema for the mock `show_buttons` UI tool."""

    prompt: str = Field(description="Instruction or question shown above the buttons")
    buttons: List[str] = Field(description="The text labels for each button")


class ShowNotificationInput(BaseModel):
    """Schema for the mock `show_notification` UI tool."""

    message: str = Field(description="Notification text to surface to the user")
    severity: str = Field(default="info", description="Level: info | success | warning | error")


class ShowProgressBarInput(BaseModel):
    """Schema for the mock `show_progress_bar` UI tool."""

    title: str = Field(description="Title of the operation being tracked, e.g. 'Uploading' ")
    percent: int = Field(ge=0, le=100, description="Progress percentage (0‑100)")


class ShowInputsInput(BaseModel):
    """Schema for the mock `show_inputs` UI tool."""
    
    prompt: str = Field(description="Instructions shown above the input form")
    inputs: List[Dict[str, Any]] = Field(description="List of input field definitions")
    submit_label: str = Field(default="Submit", description="Label for the submit button")


class ShowFileCardInput(BaseModel):
    """Schema for the mock `show_file_card` UI tool."""
    
    file_id: str = Field(description="Unique identifier for the file")
    title: str = Field(description="Display title for the file")
    thumbnail: Optional[str] = Field(default=None, description="URL to thumbnail image")
    summary: str = Field(description="Brief description of the file content")
    status: str = Field(default="ready", description="Status: ready | processing | error")


# -----------------------------------------------------------------------------
# Tool factory
# -----------------------------------------------------------------------------

def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """Return the list of tools available to the agent."""

    tools: List[BaseTool] = []

    # ------------------------------------------------------------------
    # Core math helpers
    # ------------------------------------------------------------------
    def multiply_numbers(a: float, b: float) -> str:
        """Simple multiplier — returns the product as a string."""

        return f"The result of {a} × {b} is {a * b}"

    tools.append(
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description=(
                "Multiply two numbers together. Use this when the user asks for a direct multiplication of two "
                "specific values."
            ),
            args_schema=MultiplicationInput,
        )
    )

    # ------------------------------------------------------------------
    def calculate(expression: str) -> str:
        """Evaluate a basic arithmetic expression (very limited / safe eval)."""

        allowed_chars = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed_chars):
            return "Error: Expression contains invalid characters"
        try:
            result = eval(expression)  # nosec — guarded by whitelist above
            return f"The result of {expression} is {result}"
        except Exception as exc:  # pylint: disable=broad-except
            return f"Error calculating expression: {exc}"

    tools.append(
        StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description="Calculate a simple arithmetic expression like '3 * 4'.",
            args_schema=CalculateInput,
        )
    )

    # ------------------------------------------------------------------
    # Mock UI helpers (fixed responses)
    # ------------------------------------------------------------------
    def show_buttons(prompt: str, buttons: List[str]) -> Dict[str, Any]:
        """Return a JSON payload representing a button group."""

        return {
            "kind": "buttons",
            "prompt": prompt,
            "buttons": [{"label": btn, "action": f"button_{i}"} for i, btn in enumerate(buttons)],
        }

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Display interactive buttons for user selection. Use when you want the user to choose from predefined options.",
            args_schema=ShowButtonsInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------------------
    def show_inputs(prompt: str, inputs: List[Dict[str, Any]], submit_label: str = "Submit") -> Dict[str, Any]:
        """Return a JSON payload representing an input form."""

        return {
            "kind": "inputs",
            "prompt": prompt,
            "inputs": inputs,
            "submit_label": submit_label,
        }

    tools.append(
        StructuredTool.from_function(
            func=show_inputs,
            name="show_inputs",
            description="Display a form with input fields for collecting structured data from the user.",
            args_schema=ShowInputsInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------------------
    def show_file_card(file_id: str, title: str, summary: str, thumbnail: Optional[str] = None, status: str = "ready") -> Dict[str, Any]:
        """Return a JSON payload representing a file card."""

        result = {
            "kind": "file_card",
            "file_id": file_id,
            "title": title,
            "summary": summary,
            "status": status,
            "actions": [
                {"label": "View Details", "action": "view_file_details"},
                {"label": "Download", "action": "download_file"},
            ]
        }
        
        if thumbnail:
            result["thumbnail"] = thumbnail
            
        return result

    tools.append(
        StructuredTool.from_function(
            func=show_file_card,
            name="show_file_card",
            description="Display a rich file preview card with actions. Use when presenting file information to users.",
            args_schema=ShowFileCardInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------------------
    # Additional DB‑backed tools could be appended here using `db_client`
    # ------------------------------------------------------------------

    return tools