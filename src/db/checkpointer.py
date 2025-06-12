# backend/src/db/checkpointer.py
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.agent.agent_state import AgentState, WorkflowStatus, CurrentDocumentInWorkflow, FieldMetadata
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

        # ---------- keep the ID that lives in the table ----------
        raw_data: Dict[str, Any] = record["state_data"] or {}
        raw_data["conversation_id"] = record["conversation_id"]
        # ----------------------------------------------------------

        return _checkpoint_data_to_agent_state(raw_data)

    except Exception as e:
        print(f"[load_agent_state] Error: {e}")
        return None


def save_agent_state(db_client: SupabaseClient, conversation_id: str, state: AgentState) -> bool:
    """
    Serialize AgentState Pydantic model into a JSON-ready dict, then insert/update Supabase.
    Also saves the current_document_in_workflow_state to the workflows table.
    Returns True on success.
    """
    try:
        # ------------------------------------------------------------------
        # Guarantee we have a valid UUID to use in the SQL WHERE clause.
        # If the caller passed an empty string, pull it from `state`.
        # ------------------------------------------------------------------
        if not conversation_id:
            conversation_id = state.conversation_id or ""
        if not conversation_id:               # still empty → bail early
            raise ValueError("conversation_id missing when saving AgentState")
        
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

        # Save agent_state table
        existing_agent = (
            db_client.client
            .table("agent_state")
            .select("id")
            .eq("conversation_id", conversation_id)
            .execute()
        )

        if existing_agent.data:
            # Update existing agent_state record
            agent_res = (
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
            # Insert a new agent_state row
            agent_res = (
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

        # Save workflow state if current_document_in_workflow_state exists
        if state.current_document_in_workflow_state and state.current_document_in_workflow_state.template_id:
            workflow_success = _save_workflow_state(db_client, conversation_id, state.current_document_in_workflow_state)
            if not workflow_success:
                print(f"[save_agent_state] Warning: Failed to save workflow state for conversation {conversation_id}")

        return bool(agent_res.data)

    except Exception as e:
        print(f"[save_agent_state] Error: {e}")
        return False


def _save_workflow_state(
    db_client: SupabaseClient, 
    conversation_id: str, 
    current_doc: CurrentDocumentInWorkflow
) -> bool:
    """
    Internal helper to save workflow state to workflows table.
    Uses upsert pattern: updates existing record if found, otherwise creates new one.
    """
    try:
        # Serialize fields to JSON-compatible format
        serialized_fields = {}
        for field_name, field_metadata in current_doc.fields.items():
            if isinstance(field_metadata, FieldMetadata):
                serialized_fields[field_name] = field_metadata.model_dump()
            elif isinstance(field_metadata, dict):
                # Already serialized
                serialized_fields[field_name] = field_metadata
            else:
                # Fallback for unexpected types
                serialized_fields[field_name] = {
                    "value": field_metadata,
                    "value_status": "pending",
                    "translated_value": None,
                    "translated_status": "pending"
                }

        # Prepare the workflow data
        workflow_data = {
            "file_id": current_doc.file_id or None,
            "template_id": current_doc.template_id or None,
            "conversation_id": conversation_id,
            "fields": serialized_fields,
            "base_file_public_url": current_doc.base_file_public_url or None,
            "template_file_public_url": current_doc.template_file_public_url or None,
            "template_required_fields": current_doc.template_required_fields or {},
            "translate_to": current_doc.translate_to or None,
            "current_document_version_public_url": current_doc.current_document_version_public_url or None,
        }
        
        # Check if workflow already exists for this conversation
        existing_workflow = (
            db_client.client
            .table("workflows")
            .select("id")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        
        if existing_workflow.data:
            # Update existing workflow record
            workflow_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = (
                db_client.client
                .table("workflows")
                .update(workflow_data)
                .eq("conversation_id", conversation_id)
                .execute()
            )
        else:
            # Insert new workflow record
            workflow_data["created_at"] = datetime.now(timezone.utc).isoformat()
            workflow_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = (
                db_client.client
                .table("workflows")
                .insert(workflow_data)
                .execute()
            )
        
        return bool(result.data)
        
    except Exception as e:
        print(f"[_save_workflow_state] Error: {e}")
        return False


def load_workflow_by_conversation(
    db_client: SupabaseClient, 
    conversation_id: str
) -> Optional[CurrentDocumentInWorkflow]:
    """
    Load the workflow state for a specific conversation ID.
    Returns CurrentDocumentInWorkflow object or None if not found.
    """
    try:
        result = (
            db_client.client
            .table("workflows")
            .select("*")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        
        if not result.data:
            return None
        
        record = result.data[0]  # Should only be one record per conversation
        
        # Convert fields from database format back to FieldMetadata objects
        fields = {}
        raw_fields = record.get("fields", {})
        if isinstance(raw_fields, dict):
            for field_name, field_data in raw_fields.items():
                if isinstance(field_data, dict):
                    fields[field_name] = FieldMetadata(**field_data)
                else:
                    # Fallback for simple values
                    fields[field_name] = FieldMetadata(
                        value=field_data,
                        value_status="pending",
                        translated_value=None,
                        translated_status="pending"
                    )
        
        # Convert database record back to CurrentDocumentInWorkflow
        return CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_required_fields=record.get("template_required_fields", {}),
            fields=fields,
            translate_to=record.get("translate_to", None),
            current_document_version_public_url=record.get("current_document_version_public_url", ""),
        )
        
    except Exception as e:
        print(f"[load_workflow_by_conversation] Error: {e}")
        return None


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
    
    # Serialize CurrentDocumentInWorkflow fields properly
    if state.current_document_in_workflow_state and state.current_document_in_workflow_state.fields:
        serialized_fields = {}
        for field_name, field_metadata in state.current_document_in_workflow_state.fields.items():
            if isinstance(field_metadata, FieldMetadata):
                serialized_fields[field_name] = field_metadata.model_dump()
            elif isinstance(field_metadata, dict):
                serialized_fields[field_name] = field_metadata
            else:
                # Fallback
                serialized_fields[field_name] = {
                    "value": field_metadata,
                    "value_status": "pending",
                    "translated_value": None,
                    "translated_status": "pending"
                }
        data["current_document_in_workflow_state"]["fields"] = serialized_fields
    
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
    model_data.setdefault("conversation_id", data.get("conversation_id", ""))
    model_data["messages"] = messages
    model_data["conversation_history"] = conversation_history
    model_data["workflow_status"] = wf_status

    # Handle CurrentDocumentInWorkflow fields deserialization
    if "current_document_in_workflow_state" in model_data:
        current_doc_data = model_data["current_document_in_workflow_state"]
        if isinstance(current_doc_data, dict) and "fields" in current_doc_data:
            fields = {}
            raw_fields = current_doc_data["fields"]
            if isinstance(raw_fields, dict):
                for field_name, field_data in raw_fields.items():
                    if isinstance(field_data, dict):
                        fields[field_name] = FieldMetadata(**field_data)
                    else:
                        # Fallback for simple values
                        fields[field_name] = FieldMetadata(
                            value=field_data,
                            value_status="pending",
                            translated_value=None,
                            translated_status="pending"
                        )
            current_doc_data["fields"] = fields

    # Use Pydantic's model validation to create the AgentState
    return AgentState(**model_data)