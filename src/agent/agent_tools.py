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
            "buttons": buttons,
        }

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Render a prompt with multiple buttons (labels only, no callbacks).",
            args_schema=ShowButtonsInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------------------
    def show_notification(message: str, severity: str = "info") -> Dict[str, str]:
        """Return a mock notification object."""

        return {
            "kind": "notification",
            "severity": severity,
            "message": message,
        }

    tools.append(
        StructuredTool.from_function(
            func=show_notification,
            name="show_notification",
            description="Display a toast/alert notification to the user (mock).",
            args_schema=ShowNotificationInput,
        )
    )

    # ------------------------------------------------------------------
    def show_progress_bar(title: str, percent: int) -> Dict[str, Any]:
        """Return a mock progress‑bar representation."""

        return {
            "kind": "progress_bar",
            "title": title,
            "percent": percent,
        }

    tools.append(
        StructuredTool.from_function(
            func=show_progress_bar,
            name="show_progress_bar",
            description="Show a determinate progress bar (mock).",
            args_schema=ShowProgressBarInput,
        )
    )

    # ------------------------------------------------------------------
    # Additional DB‑backed tools could be appended here using `db_client`
    # ------------------------------------------------------------------

    return tools
