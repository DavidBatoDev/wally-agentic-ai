from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.agent.agent_state import AgentState, WorkflowStatus
from src.db.db_client import SupabaseClient
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.schema.messages import BaseMessage


def load_agent_state(db_client: SupabaseClient, conversation_id: str) -> Optional[AgentState]:
    """
    Fetch the AgentState JSON from Supabase, convert it back to an AgentState dataclass.
    If no row exists, return None.
    """
    try:
        result = (
            db_client.client
            .table("agent_state")
            .select("*")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        if not result.data:
            return None

        record = result.data[0]
        raw_data: Dict[str, Any] = record["state_data"]  # JSON‐serializable dict

        return _checkpoint_data_to_agent_state(raw_data)

    except Exception as e:
        print(f"[load_agent_state] Error: {e}")
        return None


def save_agent_state(db_client: SupabaseClient, conversation_id: str, state: AgentState) -> bool:
    """
    Serialize AgentState into a JSON-ready dict, then insert/update Supabase.
    Returns True on success.
    """
    try:
        state_data: Dict[str, Any] = _agent_state_to_checkpoint_data(state)
        status = state.workflow_status.value if isinstance(state.workflow_status, WorkflowStatus) else state.workflow_status
        steps_done = state.steps_done

        existing = (
            db_client.client
            .table("agent_state")
            .select("id")
            .eq("conversation_id", conversation_id)
            .execute()
        )

        if existing.data:
            # Update existing record
            res = (
                db_client.client
                .table("agent_state")
                .update({
                    "status": status,
                    "steps_done": steps_done,
                    "state_data": state_data,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .eq("conversation_id", conversation_id)
                .execute()
            )
        else:
            # Insert a new row
            res = (
                db_client.client
                .table("agent_state")
                .insert({
                    "conversation_id": conversation_id,
                    "status": status,
                    "steps_done": steps_done,
                    "state_data": state_data,
                })
                .execute()
            )

        return bool(res.data)

    except Exception as e:
        print(f"[save_agent_state] Error: {e}")
        return False


def _agent_state_to_checkpoint_data(state: AgentState) -> Dict[str, Any]:
    """
    Convert AgentState dataclass into a JSON-serializable dict.
    """
    serialized_messages = []
    for msg in state.messages:
        base = {"type": msg.__class__.__name__, "content": msg.content}
        if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
            base["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id"):
            base["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name"):
            base["name"] = msg.name
        serialized_messages.append(base)

    serialized_history = []
    for msg in state.conversation_history:
        base = {"type": msg.__class__.__name__, "content": msg.content}
        if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
            base["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id"):
            base["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name"):
            base["name"] = msg.name
        serialized_history.append(base)

    return {
        "messages": serialized_messages,
        "conversation_history": serialized_history,
        "conversation_id": state.conversation_id,
        "user_id": state.user_id,
        "workflow_status": state.workflow_status.value if isinstance(state.workflow_status, WorkflowStatus) else state.workflow_status,
        "context": state.context,
        "user_upload_id": state.user_upload_id,
        "user_upload": state.user_upload,
        "extracted_required_fields": state.extracted_required_fields,
        "filled_required_fields": state.filled_required_fields,
        "translated_required_fields": state.translated_required_fields,
        "missing_required_fields": state.missing_required_fields,
        "translate_from": state.translate_from,
        "translate_to": state.translate_to,
        "template_id": state.template_id,
        "template_required_fields": state.template_required_fields,
        "document_version_id": state.document_version_id,
        "steps_done": state.steps_done,
    }


def _checkpoint_data_to_agent_state(data: Dict[str, Any]) -> AgentState:
    """
    Convert a JSON-serializable dict (or LangGraph’s AddableValuesDict) back into an AgentState.
    This helper will accept either plain dicts or real BaseMessage instances in data["messages"].
    """
    def _ensure_message(obj) -> BaseMessage:
        """
        If obj is already a BaseMessage, return it unchanged.
        Otherwise, build a new message from the dict.
        """
        if isinstance(obj, BaseMessage):
            return obj

        msg_type = obj.get("type", "HumanMessage")
        content = obj.get("content", "")

        if msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "AIMessage":
            m = AIMessage(content=content)
            if "tool_calls" in obj:
                m.tool_calls = obj["tool_calls"]
            return m
        elif msg_type == "ToolMessage":
            return ToolMessage(content=content, tool_call_id=obj.get("tool_call_id", ""))
        elif msg_type == "SystemMessage":
            return SystemMessage(content=content)
        else:
            return HumanMessage(content=content)

    # Deserialize the "messages" list (might already contain BaseMessage objects)
    messages = [_ensure_message(x) for x in data.get("messages", [])]

    # Deserialize the "conversation_history" list similarly
    conversation_history = [_ensure_message(x) for x in data.get("conversation_history", [])]

    # Reconstruct workflow status (string → WorkflowStatus enum)
    wf_status = data.get("workflow_status", "pending")
    if isinstance(wf_status, str):
        wf_status = WorkflowStatus(wf_status)

    return AgentState(
        messages=messages,
        conversation_history=conversation_history,
        conversation_id=data.get("conversation_id", ""),
        user_id=data.get("user_id", ""),
        workflow_status=wf_status,
        context=data.get("context", {}),
        user_upload_id=data.get("user_upload_id", ""),
        user_upload=data.get("user_upload", {}),
        extracted_required_fields=data.get("extracted_required_fields", {}),
        filled_required_fields=data.get("filled_required_fields", {}),
        translated_required_fields=data.get("translated_required_fields", {}),
        missing_required_fields=data.get("missing_required_fields", {}),
        translate_from=data.get("translate_from", ""),
        translate_to=data.get("translate_to", ""),
        template_id=data.get("template_id", ""),
        template_required_fields=data.get("template_required_fields", {}),
        document_version_id=data.get("document_version_id", ""),
        steps_done=data.get("steps_done", []),
    )
