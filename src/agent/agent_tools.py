# backend/src/agent/agent_tools.py
"""
Enhanced tools available to the LangGraph agent for performing various actions.
Re-checked for LangGraph v0.0.x compatibility and the orchestration pattern
used in LangGraphOrchestrator.

Changes in this revision
────────────────────────
✓ Fixed a trailing-comma bug that turned `upload_endpoint` into a tuple.
✓ Added structured logging (replace `print()` with `logger.debug()`).
✓ Kept return values as *JSON strings* because the orchestration layer
  expects to call `json.loads()` on the message content.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.db.db_client import SupabaseClient

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────── global state
_TOOL_LIST: List[BaseTool] | None = None
_TOOL_MAP: Dict[str, BaseTool] | None = None

# ───────────────────────────────────────────────────────── exceptions
class ToolExecutionError(Exception):
    """Raised when a tool fails internally."""


def _build_tools(db_client: Optional[SupabaseClient] = None) -> None:
    """
    Lazily (re)build the shared tool registry.

    Keeping a single in-process cache avoids recreating LangChain
    objects every time the orchestrator asks for a tool.
    """
    global _TOOL_LIST, _TOOL_MAP
    _TOOL_LIST = _TOOL_LIST or get_tools(db_client=db_client)
    _TOOL_MAP = _TOOL_MAP or {t.name: t for t in _TOOL_LIST}


# ───────────────────────────────────────────────────────── public helpers
def TOOL_MAP(db_client: Optional[SupabaseClient] = None) -> Dict[str, BaseTool]:
    """
    Access **all** tools as a name->tool dictionary.
    Example:
        chosen_tool = TOOL_MAP()[\"show_upload_button\"]
        result_json = chosen_tool.run({...})
    """
    _build_tools(db_client)
    # mypy/pyright friendly: reveal it's never None at this point
    return _TOOL_MAP or {}


def get_tool(
    name: str,
    *,
    db_client: Optional[SupabaseClient] = None,
    raise_if_missing: bool = True,
) -> Optional[BaseTool]:
    """
    Retrieve a single tool by name.

    Args:
        name:     LangGraph/LangChain tool name (e.g. \"calculate\").
        raise_if_missing:  If True, raise a KeyError when not found;
                           otherwise return None.
    """
    tools = TOOL_MAP(db_client)
    if name in tools:
        return tools[name]

    if raise_if_missing:
        raise KeyError(f"Tool '{name}' is not registered.")
    return None

# ───────────────────────────────────────────────────────── pydantic models
class MultiplicationInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class CalculateInput(BaseModel):
    expression: str = Field(
        description="Math expression to evaluate, e.g. '(3 + 4) * 5'"
    )


class ShowButtonsInput(BaseModel):
    prompt: str
    buttons: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None


class ShowUploadButtonInput(BaseModel):
    prompt: str
    accepted_types: Optional[List[str]] = None
    max_size_mb: int = 10
    multiple: bool = False
    upload_endpoint: Optional[str] = Field(
        default="/api/uploads/file",
        description="API endpoint for file upload",
    )
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ShowFileCardInput(BaseModel):
    file_id: str
    title: str
    summary: str
    thumbnail: Optional[str] = None
    status: str = "ready"
    file_size: Optional[str] = None
    file_type: Optional[str] = None



# ───────────────────────────────────────────────────────── tool factory
def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """Return all tools available to the LangGraph agent."""
    tools: List[BaseTool] = []

    # ------------------------------------------------------ math tools
    def multiply_numbers(a: float, b: float) -> str:
        try:
            result = a * b
            logger.debug("multiply_numbers %s × %s = %s", a, b, result)
            return f"The result of {a} × {b} is {result}"
        except Exception as exc:
            raise ToolExecutionError(f"Multiplication failed: {exc}") from exc

    tools.append(
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together.",
            args_schema=MultiplicationInput,
        )
    )

    def calculate(expression: str) -> str:
        allowed_chars = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed_chars):
            raise ToolExecutionError("Expression contains invalid characters")

        try:
            logger.debug("calculate expression=%s", expression)
            result = eval(expression)  # nosec — safe because of whitelist
            return f"The result of {expression} is {result}"
        except Exception as exc:
            raise ToolExecutionError(f"Calculation failed: {exc}") from exc

    tools.append(
        StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description="Safely evaluate an arithmetic expression.",
            args_schema=CalculateInput,
        )
    )

    # ------------------------------------------------------ UI helpers
    def show_buttons(
        prompt: str,
        buttons: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        logger.debug("show_buttons prompt=%s", prompt)

        formatted: List[Dict[str, str]] = []
        for i, btn in enumerate(buttons):
            if isinstance(btn, str):
                value = btn.lower().replace(" ", "_")
                formatted.append(
                    {
                        "label": btn,
                        "action": f"button_{value}",
                        "value": value,
                        "text": btn,
                    }
                )
            else:
                label = btn.get("label", f"Button {i}")
                value = btn.get("value", label.lower().replace(" ", "_"))
                formatted.append(
                    {
                        "label": label,
                        "action": btn.get("action", f"button_{value}"),
                        "value": value,
                        "text": btn.get("text", label),
                    }
                )

        payload: Dict[str, Any] = {
            "kind": "buttons",
            "prompt": prompt,
            "buttons": formatted,
            "instructions": "Click a button to continue",
        }
        if context:
            payload["context"] = context

        return json.dumps(payload)

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Render interactive buttons.",
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
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        logger.debug("show_upload_button prompt=%s", prompt)

        if accepted_types is None:
            accepted_types = [
                "image/jpeg",
                "image/png",
                "image/gif",
                "application/pdf",
            ]

        payload: Dict[str, Any] = {
            "kind": "upload_button",
            "prompt": prompt,
            "config": {
                "accepted_types": accepted_types,
                "max_size_mb": max_size_mb,
                "multiple": multiple,
                "upload_endpoint": upload_endpoint,
                "method": "POST",
                "headers": {"Content-Type": "multipart/form-data"},
                "conversation_id": conversation_id,
            },
            "ui": {
                "button_text": "Choose Files" if multiple else "Choose File",
                "drag_drop_text": f"Drag & drop {'files' if multiple else 'a file'} here or click to browse",
                "max_size_text": f"Maximum file size: {max_size_mb} MB",
                "accepted_types_text": "Accepted: "
                + ", ".join(t.split("/")[-1].upper() for t in accepted_types),
            },
        }
        if context:
            payload["context"] = context
        return json.dumps(payload)

    tools.append(
        StructuredTool.from_function(
            func=show_upload_button,
            name="show_upload_button",
            description="Render a file-upload widget.",
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
        file_type: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "kind": "file_card",
            "file_id": file_id,
            "title": title,
            "summary": summary,
            "status": status,
        }
        if thumbnail:
            payload["thumbnail"] = thumbnail
        if file_size:
            payload["file_size"] = file_size
        if file_type:
            payload["file_type"] = file_type
        return json.dumps(payload)

    tools.append(
        StructuredTool.from_function(
            func=show_file_card,
            name="show_file_card",
            description="Render a file preview card.",
            args_schema=ShowFileCardInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------ return
    return tools
