# backend/src/routers/workflow.py
"""
Workflow API router for managing workflow states and operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel, UUID4
import json

from ..dependencies import get_current_user
from ..models.user import User
from ..db.db_client import supabase_client
from ..db.workflows_db import (
    load_workflow_by_conversation, 
    get_workflow_template_mappings_by_conversation,
    get_workflow_with_template_mappings_by_conversation
)

router = APIRouter()


# ────────────────────────────────────────────────── payload models
class WorkflowResponse(BaseModel):
    success: bool
    workflow: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    conversation_id: UUID4
    has_workflow: bool
    workflow_data: Optional[Dict[str, Any]] = None


class FieldMetadataDict(BaseModel):
    value: Any
    value_status: str = "pending"
    translated_value: Optional[str] = None
    translated_status: str = "pending"


class UpdateWorkflowFieldsRequest(BaseModel):
    fields: Dict[str, FieldMetadataDict]


class UpdateSingleFieldRequest(BaseModel):
    field_key: str
    value: str
    value_status: str = "manual"
    translated_value: Optional[str] = None
    translated_status: str = "pending"


# ────────────────────────────────────────────────── helpers
def _guard_membership(conversation_id: str, current_user: User):
    """Ensure user has access to the conversation."""
    convo = supabase_client.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if convo.get("profile_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this conversation",
        )
    return convo


def _serialize_workflow(workflow_obj, template_mappings=None) -> Dict[str, Any]:
    """Convert CurrentDocumentInWorkflow object to dictionary."""
    if not workflow_obj:
        return {}
    
    workflow_dict = {
        "file_id": workflow_obj.file_id,
        "base_file_public_url": workflow_obj.base_file_public_url,
        "template_id": workflow_obj.template_id,
        "template_file_public_url": workflow_obj.template_file_public_url,
        "template_required_fields": workflow_obj.template_required_fields,
        "fields": workflow_obj.fields,
        "translate_to": workflow_obj.translate_to,
        "current_document_version_public_url": workflow_obj.current_document_version_public_url,
    }
    
    # Include origin_template_mappings if provided
    if template_mappings:
        workflow_dict["origin_template_mappings"] = template_mappings
    
    return workflow_dict


def _serialize_fields_for_db(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Convert FieldMetadata objects to JSON-serializable format for database storage."""
    serialized = {}
    for field_name, field_data in fields.items():
        if hasattr(field_data, 'model_dump'):  # Pydantic model
            serialized[field_name] = field_data.model_dump()
        elif isinstance(field_data, dict):
            serialized[field_name] = field_data
        else:
            # Fallback for simple values
            serialized[field_name] = {
                "value": field_data,
                "value_status": "pending",
                "translated_value": None,
                "translated_status": "pending"
            }
    return serialized


# ────────────────────────────────────────────────── routes
@router.get("/{conversation_id}", response_model=WorkflowStatusResponse)
async def get_workflow_by_conversation(
    conversation_id: UUID4,
    current_user: User = Depends(get_current_user),
) -> WorkflowStatusResponse:
    """
    Get the workflow state for a specific conversation.
    Returns workflow data if it exists, or indicates no workflow found.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load workflow state ------------------------------------------------
        workflow, template_mappings = get_workflow_with_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )
        
        if workflow:
            workflow_data = _serialize_workflow(workflow, template_mappings)
            return WorkflowStatusResponse(
                conversation_id=conversation_id,
                has_workflow=True,
                workflow_data=workflow_data
            )
        else:
            return WorkflowStatusResponse(
                conversation_id=conversation_id,
                has_workflow=False,
                workflow_data=None
            )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting workflow for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {exc}",
        )


@router.get("/{conversation_id}/fields", response_model=Dict[str, Any])
async def get_workflow_fields(
    conversation_id: UUID4,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get only the fields-related data from a conversation's workflow.
    Useful for form rendering and field management.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load workflow state ------------------------------------------------
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            return {
                "success": True,
                "has_workflow": False,
                "fields": {}
            }

        # Get template mappings separately
        template_mappings = get_workflow_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )

        response_data = {
            "success": True,
            "has_workflow": True,
            "fields": {
                "template_required_fields": workflow.template_required_fields or {},
                "fields": workflow.fields or {},
            }
        }
        
        # Include origin_template_mappings if available
        if template_mappings:
            response_data["fields"]["origin_template_mappings"] = template_mappings
        
        return response_data

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting workflow fields for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow fields: {exc}",
        )


@router.get("/{conversation_id}/template-mappings", response_model=Dict[str, Any])
async def get_workflow_template_mappings(
    conversation_id: UUID4,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get only the template mappings data from a conversation's workflow.
    Useful for understanding field positioning and formatting.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load workflow state ------------------------------------------------
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            return {
                "success": True,
                "has_workflow": False,
                "template_mappings": {}
            }

        # Get template mappings separately
        template_mappings = get_workflow_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        ) or {}

        return {
            "success": True,
            "has_workflow": True,
            "template_mappings": template_mappings
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting template mappings for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template mappings: {exc}",
        )


@router.patch("/{conversation_id}/field", response_model=Dict[str, Any])
async def update_single_workflow_field(
    conversation_id: UUID4,
    request: UpdateSingleFieldRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Update a single field in a conversation's workflow.
    This is used for individual field updates from the DocumentCanvas.
    Updates both workflows table and agent_state table.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load existing workflow ---------------------------------------------- 
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No workflow found for this conversation"
            )

        # ── get current fields and update the specific field -------------------
        current_fields = workflow.fields or {}
        
        # Convert FieldMetadata objects to serializable dictionaries
        serialized_current_fields = {}
        for field_name, field_data in current_fields.items():
            if hasattr(field_data, 'value'):  # FieldMetadata object
                serialized_current_fields[field_name] = {
                    "value": field_data.value,
                    "value_status": field_data.value_status,
                    "translated_value": field_data.translated_value,
                    "translated_status": field_data.translated_status
                }
            elif isinstance(field_data, dict):
                serialized_current_fields[field_name] = field_data
            else:
                # Fallback for simple values
                serialized_current_fields[field_name] = {
                    "value": field_data,
                    "value_status": "pending",
                    "translated_value": None,
                    "translated_status": "pending"
                }
        
        # Update the specific field
        updated_field = {
            "value": request.value,
            "value_status": request.value_status,
            "translated_value": request.translated_value,
            "translated_status": request.translated_status
        }
        
        serialized_current_fields[request.field_key] = updated_field
        
        # ── update workflow in database -----------------------------------------
        update_data = {
            "fields": json.dumps(serialized_current_fields),
            "updated_at": "now()"
        }
        
        result = supabase_client.client.table("workflows").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow field"
            )

        # ── update agent_state table -------------------------------------------
        try:
            # Get current agent_state
            agent_state_result = supabase_client.client.table("agent_state").select("state_data").eq(
                "conversation_id", str(conversation_id)
            ).execute()
            
            if agent_state_result.data:
                current_state_data = agent_state_result.data[0].get("state_data", {})
                
                # Update the current_document_in_workflow_state.fields
                if "current_document_in_workflow_state" not in current_state_data:
                    current_state_data["current_document_in_workflow_state"] = {}
                
                if "fields" not in current_state_data["current_document_in_workflow_state"]:
                    current_state_data["current_document_in_workflow_state"]["fields"] = {}
                
                # Update the specific field in agent_state
                current_state_data["current_document_in_workflow_state"]["fields"][request.field_key] = updated_field
                
                # Update agent_state table
                agent_state_update_result = supabase_client.client.table("agent_state").update({
                    "state_data": current_state_data,
                    "updated_at": "now()"
                }).eq("conversation_id", str(conversation_id)).execute()
                
                if not agent_state_update_result.data:
                    print(f"Warning: Failed to update agent_state for conversation {conversation_id}")
            else:
                print(f"Warning: No agent_state found for conversation {conversation_id}")
                
        except Exception as agent_state_error:
            print(f"Error updating agent_state for conversation {conversation_id}: {agent_state_error}")
            # Don't fail the entire request if agent_state update fails
            # The workflow table update was successful, so we can continue

        return {
            "success": True,
            "message": f"Field '{request.field_key}' updated successfully",
            "field_key": request.field_key,
            "updated_field": updated_field,
            "all_fields": serialized_current_fields
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error updating single workflow field for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow field: {exc}",
        )

@router.patch("/{conversation_id}/fields", response_model=Dict[str, Any])
async def update_workflow_fields(
    conversation_id: UUID4,
    request: UpdateWorkflowFieldsRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Update multiple fields for a conversation's workflow.
    Updates both workflows table and agent_state table.
    """
    try:
        # ── membership / auth ---------------------------------------------------
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load existing workflow ---------------------------------------------- 
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No workflow found for this conversation"
            )

        # ── update workflow fields ---------------------------------------------- 
        # Serialize the fields for database storage
        serialized_fields = _serialize_fields_for_db(request.fields)
        
        # Update the workflow in the database
        update_data = {
            "fields": json.dumps(serialized_fields),
            "updated_at": "now()"
        }
        
        result = supabase_client.client.table("workflows").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow fields"
            )

        # ── update agent_state table -------------------------------------------
        try:
            # Get current agent_state
            agent_state_result = supabase_client.client.table("agent_state").select("state_data").eq(
                "conversation_id", str(conversation_id)
            ).execute()
            
            if agent_state_result.data:
                current_state_data = agent_state_result.data[0].get("state_data", {})
                
                # Update the current_document_in_workflow_state.fields
                if "current_document_in_workflow_state" not in current_state_data:
                    current_state_data["current_document_in_workflow_state"] = {}
                
                # Replace all fields in agent_state with the updated fields
                current_state_data["current_document_in_workflow_state"]["fields"] = serialized_fields
                
                # Update agent_state table
                agent_state_update_result = supabase_client.client.table("agent_state").update({
                    "state_data": current_state_data,
                    "updated_at": "now()"
                }).eq("conversation_id", str(conversation_id)).execute()
                
                if not agent_state_update_result.data:
                    print(f"Warning: Failed to update agent_state for conversation {conversation_id}")
            else:
                print(f"Warning: No agent_state found for conversation {conversation_id}")
                
        except Exception as agent_state_error:
            print(f"Error updating agent_state for conversation {conversation_id}: {agent_state_error}")
            # Don't fail the entire request if agent_state update fails
            # The workflow table update was successful, so we can continue

        return {
            "success": True,
            "message": "Workflow fields updated successfully",
            "updated_fields": serialized_fields
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error updating workflow fields for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow fields: {exc}",
        )