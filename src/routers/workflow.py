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
from ..db.workflows_db import load_workflow_by_conversation

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


def _serialize_workflow(workflow_obj) -> Dict[str, Any]:
    """Convert CurrentDocumentInWorkflow object to dictionary."""
    if not workflow_obj:
        return {}
    
    return {
        "file_id": workflow_obj.file_id,
        "base_file_public_url": workflow_obj.base_file_public_url,
        "template_id": workflow_obj.template_id,
        "template_file_public_url": workflow_obj.template_file_public_url,
        "template_required_fields": workflow_obj.template_required_fields,
        "extracted_fields_from_raw_ocr": workflow_obj.extracted_fields_from_raw_ocr,
        "filled_fields": workflow_obj.filled_fields,
        "translated_fields": workflow_obj.translated_fields,
    }


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
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if workflow:
            workflow_data = _serialize_workflow(workflow)
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

        return {
            "success": True,
            "has_workflow": True,
            "fields": {
                "template_required_fields": workflow.template_required_fields or {},
                "extracted_fields_from_raw_ocr": workflow.extracted_fields_from_raw_ocr or {},
                "filled_fields": workflow.filled_fields or {},
                "translated_fields": workflow.translated_fields or {},
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting workflow fields for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow fields: {exc}",
        )
    
    # Add this payload model
class UpdateWorkflowFieldsRequest(BaseModel):
    filled_fields: Dict[str, str]
    translated_fields: Dict[str, str]

# Add this route to your workflow router
@router.patch("/{conversation_id}/fields", response_model=Dict[str, Any])
async def update_workflow_fields(
    conversation_id: UUID4,
    request: UpdateWorkflowFieldsRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Update the filled_fields and translated_fields for a conversation's workflow.
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
        # Update the workflow in the database
        update_data = {
            "filled_fields": json.dumps(request.filled_fields),
            "translated_fields": json.dumps(request.translated_fields),
            "updated_at": "now()"
        }
        
        result = supabase_client.table("workflow").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow fields"
            )

        return {
            "success": True,
            "message": "Workflow fields updated successfully",
            "updated_fields": {
                "filled_fields": request.filled_fields,
                "translated_fields": request.translated_fields
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error updating workflow fields for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow fields: {exc}",
        )
