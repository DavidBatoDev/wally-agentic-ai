# backend/src/db/workflow_db.py
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json

from src.agent.agent_state import AgentState, CurrentDocumentInWorkflow, FieldMetadata
from src.db.db_client import SupabaseClient


def _extract_template_mappings(db_client: SupabaseClient, template_id: str) -> Dict[str, Any]:
    """
    Extract fillable_text_info from template.info_json and return only the required fields.
    Returns a dictionary mapping keys to their metadata (label, font, position, page_number, bbox_center, rotation, alignment).
    """
    try:
        if not template_id:
            return {}
        
        # Get template info_json
        result = (
            db_client.client
            .table("templates")
            .select("info_json")
            .eq("id", template_id)
            .execute()
        )
        
        if not result.data:
            return {}
        
        info_json = result.data[0].get("info_json", {})
        fillable_text_info = info_json.get("fillable_text_info", [])
        
        # Extract only the required fields for each fillable text item
        template_mappings = {}
        for item in fillable_text_info:
            if not isinstance(item, dict) or "key" not in item:
                continue
                
            key = item["key"]
            
            # Calculate bbox_center from position if available
            bbox_center = None
            position = item.get("position", {})
            if position and all(coord in position for coord in ["x0", "y0", "x1", "y1"]):
                bbox_center = {
                    "x": (position["x0"] + position["x1"]) / 2,
                    "y": (position["y0"] + position["y1"]) / 2
                }
            
            template_mappings[key] = {
                "label": item.get("label", ""),
                "font": item.get("font", {}),
                "position": position,
                "page_number": item.get("page_number", 1),
                "bbox_center": bbox_center,
                "rotation": item.get("rotation", 0),
                "alignment": item.get("alignment", "left")
            }
        
        return template_mappings
        
    except Exception as e:
        print(f"[_extract_template_mappings] Error extracting template mappings: {e}")
        return {}


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
        
        # Convert FieldMetadata objects to JSON-serializable format
        fields_json = {}
        for field_name, field_metadata in current_doc.fields.items():
            if isinstance(field_metadata, FieldMetadata):
                fields_json[field_name] = {
                    "value": field_metadata.value,
                    "value_status": field_metadata.value_status,
                    "translated_value": field_metadata.translated_value,
                    "translated_status": field_metadata.translated_status,
                }
            else:
                # Handle case where it might already be a dict
                fields_json[field_name] = field_metadata
        
        # Extract template mappings from template info_json
        origin_template_mappings = {}
        if current_doc.template_id:
            origin_template_mappings = _extract_template_mappings(db_client, current_doc.template_id)
        
        # Extract translated template mappings from template_translated_id
        translated_template_mappings = {}
        if hasattr(current_doc, 'template_translated_id') and current_doc.template_translated_id:
            translated_template_mappings = _extract_template_mappings(db_client, current_doc.template_translated_id)
        
        # Prepare the workflow data
        workflow_data = {
            "file_id": current_doc.file_id or None,
            "template_id": current_doc.template_id or None,
            "conversation_id": conversation_id,
            "fields": fields_json,
            "base_file_public_url": current_doc.base_file_public_url or None,
            "template_file_public_url": current_doc.template_file_public_url or None,
            "template_translated_id": getattr(current_doc, 'template_translated_id', None),
            "template_translated_file_public_url": getattr(current_doc, 'template_translated_file_public_url', None),
            "template_required_fields": current_doc.template_required_fields or {},
            "translate_to": current_doc.translate_to or None,
            "translate_from": getattr(current_doc, 'translate_from', None),
            "current_document_version_public_url": current_doc.current_document_version_public_url or None,
            "origin_template_mappings": origin_template_mappings,
            "translated_template_mappings": translated_template_mappings,
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
        
        # Convert fields JSON back to FieldMetadata objects
        fields_dict = {}
        fields_data = record.get("fields", {}) or {}
        
        # Handle case where fields might be stored as JSON string
        if isinstance(fields_data, str):
            try:
                fields_data = json.loads(fields_data)
            except json.JSONDecodeError:
                print(f"[load_workflow_state] Failed to parse fields JSON: {fields_data}")
                fields_data = {}
        
        # Ensure fields_data is a dictionary
        if not isinstance(fields_data, dict):
            fields_data = {}
        
        for field_name, field_data in fields_data.items():
            if isinstance(field_data, dict):
                fields_dict[field_name] = FieldMetadata(
                    value=field_data.get("value"),
                    value_status=field_data.get("value_status", "pending"),
                    translated_value=field_data.get("translated_value"),
                    translated_status=field_data.get("translated_status", "pending"),
                )
            else:
                # Handle legacy data or simple values
                fields_dict[field_name] = FieldMetadata(value=field_data)
        
        # Convert database record back to CurrentDocumentInWorkflow
        current_doc = CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_translated_id=record.get("template_translated_id"),
            template_translated_file_public_url=record.get("template_translated_file_public_url"),
            template_required_fields=record.get("template_required_fields", {}),
            fields=fields_dict,
            translate_to=record.get("translate_to"),
            translate_from=record.get("translate_from"),
            current_document_version_public_url=record.get("current_document_version_public_url") or "",
        )
        
        return current_doc
        
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
        
        # Convert fields JSON back to FieldMetadata objects
        fields_dict = {}
        fields_data = record.get("fields", {}) or {}
        
        # Handle case where fields might be stored as JSON string
        if isinstance(fields_data, str):
            try:
                fields_data = json.loads(fields_data)
            except json.JSONDecodeError:
                print(f"[load_workflow_by_conversation] Failed to parse fields JSON: {fields_data}")
                fields_data = {}
        
        # Ensure fields_data is a dictionary
        if not isinstance(fields_data, dict):
            fields_data = {}
        
        for field_name, field_data in fields_data.items():
            if isinstance(field_data, dict):
                fields_dict[field_name] = FieldMetadata(
                    value=field_data.get("value"),
                    value_status=field_data.get("value_status", "pending"),
                    translated_value=field_data.get("translated_value"),
                    translated_status=field_data.get("translated_status", "pending"),
                )
            else:
                # Handle legacy data or simple values
                fields_dict[field_name] = FieldMetadata(value=field_data)
        
        # Convert database record back to CurrentDocumentInWorkflow
        current_doc = CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_translated_id=record.get("template_translated_id"),
            template_translated_file_public_url=record.get("template_translated_file_public_url"),
            template_required_fields=record.get("template_required_fields", {}),
            fields=fields_dict,
            translate_to=record.get("translate_to"),
            translate_from=record.get("translate_from"),
            current_document_version_public_url=record.get("current_document_version_public_url") or "",
        )
        
        return current_doc
        
    except Exception as e:
        print(f"[load_workflow_by_conversation] Error: {e}")
        return None

def get_workflow_template_mappings_by_conversation(
    db_client: SupabaseClient, 
    conversation_id: str
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Get both origin and translated template mappings for a specific conversation ID.
    Returns a tuple of (origin_template_mappings, translated_template_mappings).
    Both can be None if not found.
    """
    try:
        result = (
            db_client.client
            .table("workflows")
            .select("origin_template_mappings, translated_template_mappings")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        
        if not result.data:
            return None, None
        
        record = result.data[0]
        origin_mappings = record.get("origin_template_mappings", {})
        translated_mappings = record.get("translated_template_mappings", {})
        
        return origin_mappings, translated_mappings
        
    except Exception as e:
        print(f"[get_workflow_template_mappings_by_conversation] Error: {e}")
        return None, None

def get_workflow_with_template_mappings_by_conversation(
    db_client: SupabaseClient, 
    conversation_id: str
) -> tuple[Optional[CurrentDocumentInWorkflow], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load the workflow state and both template mappings for a specific conversation ID.
    Returns a tuple of (workflow_state, origin_template_mappings, translated_template_mappings).
    All can be None if not found.
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
            return None, None, None
        
        record = result.data[0]
        
        # Convert fields JSON back to FieldMetadata objects
        fields_dict = {}
        fields_data = record.get("fields", {}) or {}
        
        # Handle case where fields might be stored as JSON string
        if isinstance(fields_data, str):
            try:
                fields_data = json.loads(fields_data)
            except json.JSONDecodeError:
                print(f"[get_workflow_with_template_mappings_by_conversation] Failed to parse fields JSON: {fields_data}")
                fields_data = {}
        
        # Ensure fields_data is a dictionary
        if not isinstance(fields_data, dict):
            fields_data = {}
        
        for field_name, field_data in fields_data.items():
            if isinstance(field_data, dict):
                fields_dict[field_name] = FieldMetadata(
                    value=field_data.get("value"),
                    value_status=field_data.get("value_status", "pending"),
                    translated_value=field_data.get("translated_value"),
                    translated_status=field_data.get("translated_status", "pending"),
                )
            else:
                # Handle legacy data or simple values
                fields_dict[field_name] = FieldMetadata(value=field_data)
        
        # Convert database record back to CurrentDocumentInWorkflow
        current_doc = CurrentDocumentInWorkflow(
            file_id=record.get("file_id", ""),
            base_file_public_url=record.get("base_file_public_url", ""),
            template_id=record.get("template_id", ""),
            template_file_public_url=record.get("template_file_public_url", ""),
            template_translated_id=record.get("template_translated_id"),
            template_translated_file_public_url=record.get("template_translated_file_public_url"),
            template_required_fields=record.get("template_required_fields", {}),
            fields=fields_dict,
            translate_to=record.get("translate_to"),
            translate_from=record.get("translate_from"),
            current_document_version_public_url=record.get("current_document_version_public_url") or "",
        )
        
        # Get both template mappings
        origin_template_mappings = record.get("origin_template_mappings", {})
        translated_template_mappings = record.get("translated_template_mappings", {})
        
        return current_doc, origin_template_mappings, translated_template_mappings
        
    except Exception as e:
        print(f"[get_workflow_with_template_mappings_by_conversation] Error: {e}")
        return None, None, None

def get_translated_template_mappings_by_conversation(
    db_client: SupabaseClient, 
    conversation_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get only the translated template mappings for a specific conversation ID.
    Returns the translated_template_mappings dictionary or None if not found.
    """
    try:
        result = (
            db_client.client
            .table("workflows")
            .select("translated_template_mappings")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        
        if not result.data:
            return None
        
        record = result.data[0]
        return record.get("translated_template_mappings", {})
        
    except Exception as e:
        print(f"[get_translated_template_mappings_by_conversation] Error: {e}")
        return None