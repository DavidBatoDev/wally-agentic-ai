# backend/src/agent/core_tools.py

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


class ShowUploadButtonInput(BaseModel):
    prompt: str
    accepted_types: Optional[List[str]] = None
    max_size_mb: int = 10
    multiple: bool = False
    upload_endpoint: Optional[str] = "/api/uploads/file"
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AgentStatePatch(BaseModel):
    translate_from: Optional[str] = Field(
        None, 
        description="The document language"
    )
    translate_to: Optional[str] = Field(
        None, 
        description="Target language"
    )
    conversation_id: Optional[str] = Field(
        None, 
        description="Auto-filled by orchestrator"
    )


def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """Return all tools available to the LangGraph agent."""
    tools: List[BaseTool] = []

    # ------------------------------------------------------
    # 1) update_agent_state (synchronous!)
    def update_agent_state(**patch_kwargs) -> str:
        """
        A tool to update the agent's state to identify if the user provided a new infromation about what language do they want to translate to.
        """
        print(f"[TOOL] update_agent_state called with arguments: {patch_kwargs}")
        
        bullet: List[str] = []

        if "translate_to" in patch_kwargs:
            lang = patch_kwargs["translate_to"]
            bullet.append(f"✔️ Target language set to **{lang}**.")

        if not bullet:
            bullet.append("Workflow state updated.")

        result = " ".join(bullet)
        print(f"[TOOL] update_agent_state result: {result}")

        supabase_client.client.table('messages').insert({
            "conversation_id": patch_kwargs.get("conversation_id", ""),
            "sender": "assistant",
            "kind": "text",
            "body": json.dumps({"text": result}),
        }).execute()

        return result

    tools.append(
        StructuredTool.from_function(
            func=update_agent_state,
            name="update_translate_to",
            description=(
                "Update the agent's state with new information, such as the target language for translation."
            ),
            args_schema=AgentStatePatch,
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
        if ".pdf" in accepted_types:
            accepted_types.append("application/pdf")
        if ".txt" in accepted_types:
            accepted_types.append("text/plain")
        if ".docx" in accepted_types:
            accepted_types.append("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
        accepted_types.append("image/png")
        accepted_types.append("image/jpeg")

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