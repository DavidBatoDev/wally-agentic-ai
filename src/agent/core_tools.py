# backend/src/agent/agent_tools.py

"""
Enhanced tools available to the LangGraph agent for performing various actions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.db.db_client import SupabaseClient, supabase_client

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────── global state
_TOOL_LIST: List[BaseTool] | None = None
_TOOL_MAP: Dict[str, BaseTool] | None = None

class ToolExecutionError(Exception):
    """Raised when a tool fails internally."""

def TOOL_MAP(db_client: Optional[SupabaseClient] = None) -> Dict[str, BaseTool]:
    """
    Access all tools as a name->tool dictionary.
    """
    global _TOOL_LIST, _TOOL_MAP
    if _TOOL_LIST is None:
        _TOOL_LIST = get_tools(db_client=db_client)
        _TOOL_MAP = {t.name: t for t in _TOOL_LIST}
    return _TOOL_MAP or {}

def get_tool(
    name: str,
    *,
    db_client: Optional[SupabaseClient] = None,
    raise_if_missing: bool = True,
) -> Optional[BaseTool]:
    tools = TOOL_MAP(db_client)
    if name in tools:
        return tools[name]
    if raise_if_missing:
        raise KeyError(f"Tool '{name}' is not registered.")
    return None


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
    conversation_id: Optional[str] = None

class ShowUploadButtonInput(BaseModel):
    prompt: str
    accepted_types: Optional[List[str]] = None
    max_size_mb: int = 10
    multiple: bool = False
    upload_endpoint: Optional[str] = "/api/uploads/file"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AgentStatePatch(BaseModel):
    translate_from: Optional[str] = Field(None, description="The document language")
    translate_to: Optional[str] = Field(None, description="Target language")
    user_upload_info: Union[Dict[str, Any], str, None] = Field(
        default=None,
        description=(
            "Metadata about the user-uploaded file. "
            "Prefer a dict, but a plain string is tolerated and will be parsed."
        ),
    )
    upload_analysis: Dict[str, Any] = Field(default_factory=dict)
    user_upload_id: Optional[str] = Field(
        None,
        description="Supabase UUID of the user-uploaded file",
    )
    template_id: Optional[str] = Field(None, description="The template ID to use to find the template document for translation")
    workflow_status: Optional[str] = Field(
        None,
        description="pending, in_progress, waiting_confirmation, completed",
    )
    conversation_id: Optional[str] = Field(None, description="Auto-filled by orchestrator")



def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """Return all tools available to the LangGraph agent."""
    tools: List[BaseTool] = []

    # ------------------------------------------------------
    # 1) A simple "multiply" tool
    def multiply_numbers(a: float, b: float) -> str:
        print(f"[TOOL] multiply_numbers called with arguments: a={a}, b={b}")
        try:
            result = a * b
            print(f"[TOOL] multiply_numbers result: {result}")
            logger.debug("multiply_numbers %s × %s = %s", a, b, result)
            return f"The result of {a} × {b} is {result}"
        except Exception as exc:
            print(f"[TOOL] multiply_numbers error: {exc}")
            raise ToolExecutionError(f"Multiplication failed: {exc}") from exc

    tools.append(
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together.",
            args_schema=MultiplicationInput,
        )
    )

    # ------------------------------------------------------
    # 2) update_agent_state (synchronous!)
    def update_agent_state(**patch_kwargs) -> str:
        """
        Called by the LLM to store new workflow info.
        We also craft a friendly confirmation for the user.
        """
        print(f"[TOOL] update_agent_state called with arguments: {patch_kwargs}")
        
        bullet: List[str] = []

        if "translate_to" in patch_kwargs:
            lang = patch_kwargs["translate_to"]
            bullet.append(f"✔️ Target language set to **{lang}**.")

        if "translate_from" in patch_kwargs:
            src = patch_kwargs["translate_from"]
            bullet.append(f"✔️ Source language noted as **{src}**.")

        if "user_upload" in patch_kwargs:
            upload = patch_kwargs["user_upload"]
            if "name" in upload:
                bullet.append(f"✔️ File **{upload['name']}** received.")
            else:
                bullet.append("✔️ File uploaded.")

            if "size_bytes" in upload:
                size_mb = upload["size_bytes"] / (1024 * 1024)
                bullet.append(f"✔️ File size: {size_mb:.2f} MB.")

            if "mime_type" in upload:
                bullet.append(f"✔️ File type: {upload['mime_type']}.")

        if "template_id" in patch_kwargs:
            bullet.append("✔️ Template ID received and stored.")

        if "doc_type" in patch_kwargs:
            bullet.append(f"✔️ Document type recorded: {patch_kwargs['doc_type']}.")

        if not bullet:
            bullet.append("Workflow state updated.")

        # 👉 teach the user what to do next
        if "user_upload_id" not in patch_kwargs:
            bullet.append("📄 Please upload the document whenever you're ready.")

        result = " ".join(bullet)
        print(f"[TOOL] update_agent_state result: {result}")
        return result

    tools.append(
        StructuredTool.from_function(
            func=update_agent_state,
            name="update_agent_state",
            description=(
                "Record NEW information you inferred from the user's message. "
                "Only include keys that changed."
            ),
            args_schema=AgentStatePatch,
            return_direct=False,
        )
    )

    # ------------------------------------------------------
    # 3) show_buttons
    def show_buttons(
        prompt: str,
        buttons: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        print(f"[TOOL] show_buttons called with arguments: prompt='{prompt}', buttons={buttons}, context={context}, conversation_id={conversation_id}")
        
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
        if conversation_id:
            payload["conversation_id"] = conversation_id

        result = json.dumps(payload)
        print(f"[TOOL] show_buttons result: {result}")
        return result

    tools.append(
        StructuredTool.from_function(
            func=show_buttons,
            name="show_buttons",
            description="Render interactive buttons.",
            args_schema=ShowButtonsInput,
            return_direct=True,
        )
    )

    # ------------------------------------------------------
    # 4) show_upload_button
    def show_upload_button(
        prompt: str,
        accepted_types: Optional[List[str]] = None,
        max_size_mb: int = 10,
        multiple: bool = False,
        upload_endpoint: str = "/api/uploads/file",
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        print(f"[TOOL] show_upload_button called with arguments: prompt='{prompt}', accepted_types={accepted_types}, max_size_mb={max_size_mb}, multiple={multiple}, upload_endpoint='{upload_endpoint}', conversation_id={conversation_id}, context={context}")
        
        logger.debug("show_upload_button prompt=%s", prompt)

        if accepted_types is None:
            accepted_types = ["image/jpeg", "image/png", "image/gif", "application/pdf"]

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
        
        result = json.dumps(payload)
        print(f"[TOOL] show_upload_button result: {result}")
        return result

    tools.append(
        StructuredTool.from_function(
            func=show_upload_button,
            name="show_upload_button",
            description="Render a file-upload widget.",
            args_schema=ShowUploadButtonInput,
            return_direct=True,
        )
    )

    return tools