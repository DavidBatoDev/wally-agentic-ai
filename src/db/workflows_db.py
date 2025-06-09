# backend/src/db/workflow_checkpointer.py
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.agent.agent_state import AgentState, CurrentDocumentInWorkflow
from src.db.db_client import SupabaseClient


def save_current_document_in_workflow_state(
    db_client: SupabaseClient, 
    conversation_id: str, 
    state: AgentState
) -> bool:
    """
    Save the current_document_in_workflow_state from AgentState into the workflows table.
    Uses upsert pattern: updates existing record if found, otherwise creates new one.
    Ensures only one workflow record exists per conversation_id.
    Returns True on success.
    """
    try:
        # Validate inputs
        if not conversation_id:
            conversation_id = state.conversation_id or ""
        if not conversation_id:
            raise ValueError("conversation_id missing when saving workflow state")
        
        current_doc = state.current_document_in_workflow_state
        if not current_doc:
            raise ValueError("current_document_in_workflow_state is missing from AgentState")
        
        # Prepare the workflow data
        workflow_data = {
            "file_id": current_doc.file_id or None,
            "template_id": current_doc.template_id or None,
            "conversation_id": conversation_id,
            "filled_fields": current_doc.filled_fields or {},
            "translated_fields": current_doc.translated_fields or {},
            "base_file_public_url": current_doc.base_file_public_url or None,
            "template_file_public_url": current_doc.template_file_public_url or None,
            "template_required_fields": current_doc.template_required_fields or {},
            "extracted_fields_from_raw_ocr": current_doc.extracted_fields_from_raw_ocr or {},
        }
        
        # Check if workflow already exists for this conversation
        existing = (
            db_client.client
            .table("workflows")
            .select("id")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        
        if existing.data:
            # Update existing record
            workflow_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = (
                db_client.client
                .table("workflows")
                .update(workflow_data)
                .eq("conversation_id", conversation_id)
                .execute()
            )
        else:
            # Insert new record
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
        print(f"[save_current_document_in_workflow_state] Error: {e}")
        return False


def load_workflow_state(
    db_client: SupabaseClient, 
    workflow_id: str
) -> Optional[CurrentDocumentInWorkflow]:
    """
    Load a workflow state from the workflows table by workflow ID.
    Returns CurrentDocumentInWorkflow object or None if not found.
    """
    try:
        result = (
            db_client.client
            .table("workflows")
            .select("*")
            .eq("id", workflow_id)
            .execute()
        )
        
        if not result.data:
            return None
        
        record = result.data[0]
        
        # Convert database record back to CurrentDocumentInWorkflow
        return CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_required_fields=record.get("template_required_fields", {}),
            extracted_fields_from_raw_ocr=record.get("extracted_fields_from_raw_ocr", {}),
            filled_fields=record.get("filled_fields", {}),
            translated_fields=record.get("translated_fields", {}),
        )
        
    except Exception as e:
        print(f"[load_workflow_state] Error: {e}")
        return None


def load_workflow_by_conversation(
    db_client: SupabaseClient, 
    conversation_id: str
) -> Optional[CurrentDocumentInWorkflow]:
    """
    Load the workflow state for a specific conversation ID.
    Since there's only one workflow per conversation, returns the single workflow
    or None if not found.
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
        
        # Convert database record back to CurrentDocumentInWorkflow
        return CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_required_fields=record.get("template_required_fields", {}),
            extracted_fields_from_raw_ocr=record.get("extracted_fields_from_raw_ocr", {}),
            filled_fields=record.get("filled_fields", {}),
            translated_fields=record.get("translated_fields", {}),
        )
        
    except Exception as e:
        print(f"[load_workflow_by_conversation] Error: {e}")
        return None