# backend/src/db/checkpointer.py
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.agent.agent_state import AgentState, WorkflowStatus
from src.db.db_client import SupabaseClient
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.schema.messages import BaseMessage


def load_agent_state(db_client: SupabaseClient, conversation_id: str) -> Optional[AgentState]:
    """
    Fetch the AgentState JSON from Supabase, convert it back to an AgentState Pydantic model.
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
    Serialize AgentState Pydantic model into a JSON-ready dict, then insert/update Supabase.
    Returns True on success.
    """
    try:
        # Use Pydantic's model_dump for serialization
        state_data: Dict[str, Any] = _agent_state_to_checkpoint_data(state)
        
        # Handle workflow_status safely - it might already be a string due to use_enum_values
        if isinstance(state.workflow_status, WorkflowStatus):
            status = state.workflow_status.value
        elif isinstance(state.workflow_status, str):
            status = state.workflow_status
        else:
            status = str(state.workflow_status)
            
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
    Convert AgentState Pydantic model into a JSON-serializable dict.
    """
    # Helper function to serialize messages
    def serialize_message(msg: BaseMessage) -> Dict[str, Any]:
        base = {"type": msg.__class__.__name__, "content": msg.content}
        if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
            base["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id"):
            base["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name"):
            base["name"] = msg.name
        return base

    # Get the model data as a dict
    data = state.model_dump()
    
    # Manually serialize the message objects since they're not JSON serializable
    data["messages"] = [serialize_message(msg) for msg in state.messages]
    data["conversation_history"] = [serialize_message(msg) for msg in state.conversation_history]
    
    # Ensure workflow_status is serialized as string value
    if isinstance(state.workflow_status, WorkflowStatus):
        data["workflow_status"] = state.workflow_status.value
    elif isinstance(state.workflow_status, str):
        data["workflow_status"] = state.workflow_status
    else:
        data["workflow_status"] = str(state.workflow_status)
    
    return data


def _checkpoint_data_to_agent_state(data: Dict[str, Any]) -> AgentState:
    """
    Convert a JSON-serializable dict (or LangGraph's AddableValuesDict) back into an AgentState.
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
    wf_status_raw = data.get("workflow_status", "pending")
    if isinstance(wf_status_raw, WorkflowStatus):
        wf_status = wf_status_raw
    elif isinstance(wf_status_raw, str):
        try:
            wf_status = WorkflowStatus(wf_status_raw)
        except ValueError:
            # Fallback to PENDING if invalid value
            wf_status = WorkflowStatus.PENDING
    else:
        wf_status = WorkflowStatus.PENDING

    # Create a copy of data for Pydantic model creation
    model_data = data.copy()
    model_data["messages"] = messages
    model_data["conversation_history"] = conversation_history
    model_data["workflow_status"] = wf_status

    # Use Pydantic's model validation to create the AgentState
    return AgentState(**model_data)