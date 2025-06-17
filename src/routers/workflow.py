# backend/src/routers/workflow.py
"""
Workflow API router for managing workflow states and operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, UUID4, Field
import json
import io
import fitz
from fitz import Rect, TEXT_ALIGN_CENTER, TEXT_ALIGN_RIGHT, TEXT_ALIGN_JUSTIFY, TEXT_ALIGN_LEFT
import os
import httpx
from fastapi.responses import StreamingResponse

from ..dependencies import get_current_user
from ..models.user import User
from ..db.db_client import supabase_client
from ..db.workflows_db import (
    load_workflow_by_conversation, 
    get_workflow_template_mappings_by_conversation,
    get_workflow_with_template_mappings_by_conversation,
    get_translated_template_mappings_by_conversation
)
from ..services.translation_service import translation_service

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

# ────────────────────────────────────────────────── Translation Service
class TranslateAllFieldsRequest(BaseModel):
    target_language: str
    source_language: Optional[str] = None
    use_gemini: bool = True
    force_retranslate: bool = False  # If True, retranslate even if status is 'edited' or 'confirmed'


class TranslateSingleFieldRequest(BaseModel):
    field_key: str
    target_language: str
    source_language: Optional[str] = None
    use_gemini: bool = True


class TranslationResponse(BaseModel):
    success: bool
    message: str
    translated_fields: Dict[str, Any] = {}
    skipped_fields: Dict[str, str] = {}  # field_key -> reason for skipping
    errors: Dict[str, str] = {}  # field_key -> error message

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


def _serialize_workflow(workflow_obj, origin_template_mappings=None, translated_template_mappings=None) -> Dict[str, Any]:
    """Convert CurrentDocumentInWorkflow object to dictionary with template mappings."""
    if not workflow_obj:
        return {}
    
    workflow_dict = {
        "file_id": workflow_obj.file_id,
        "base_file_public_url": workflow_obj.base_file_public_url,
        "template_id": workflow_obj.template_id,
        "template_file_public_url": workflow_obj.template_file_public_url,
        "template_translated_id": getattr(workflow_obj, 'template_translated_id', None),
        "template_translated_file_public_url": getattr(workflow_obj, 'template_translated_file_public_url', None),
        "template_required_fields": workflow_obj.template_required_fields,
        "fields": workflow_obj.fields,
        "translate_to": workflow_obj.translate_to,
        "translate_from": getattr(workflow_obj, 'translate_from', None),
        "current_document_version_public_url": workflow_obj.current_document_version_public_url,
    }
    
    # Include both template mappings if provided
    if origin_template_mappings:
        workflow_dict["origin_template_mappings"] = origin_template_mappings
    
    if translated_template_mappings:
        workflow_dict["translated_template_mappings"] = translated_template_mappings
    
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
        workflow, origin_template_mappings, translated_template_mappings = get_workflow_with_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )
        
        if workflow:
            workflow_data = _serialize_workflow(workflow, origin_template_mappings, translated_template_mappings)
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
        workflow, origin_template_mappings, translated_template_mappings = get_workflow_with_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )
        
        if not workflow:
            return {
                "success": True,
                "has_workflow": False,
                "fields": {}
            }

        response_data = {
            "success": True,
            "has_workflow": True,
            "fields": {
                "template_required_fields": workflow.template_required_fields or {},
                "fields": workflow.fields or {},
            }
        }
        
        # Include both template mappings if available
        if origin_template_mappings:
            response_data["fields"]["origin_template_mappings"] = origin_template_mappings
        
        if translated_template_mappings:
            response_data["fields"]["translated_template_mappings"] = translated_template_mappings
        
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
    Get both origin and translated template mappings data from a conversation's workflow.
    Useful for understanding field positioning and formatting for both templates.
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
                "origin_template_mappings": {},
                "translated_template_mappings": {}
            }

        # Get both template mappings
        origin_template_mappings, translated_template_mappings = get_workflow_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )

        return {
            "success": True,
            "has_workflow": True,
            "origin_template_mappings": origin_template_mappings or {},
            "translated_template_mappings": translated_template_mappings or {}
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting template mappings for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template mappings: {exc}",
        )


@router.get("/{conversation_id}/translated-template-mappings", response_model=Dict[str, Any])
async def get_workflow_translated_template_mappings(
    conversation_id: UUID4,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get only the translated template mappings data from a conversation's workflow.
    Useful for understanding field positioning and formatting for the translated template.
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
                "translated_template_mappings": {}
            }

        # Get translated template mappings
        translated_template_mappings = get_translated_template_mappings_by_conversation(
            supabase_client, str(conversation_id)
        )

        return {
            "success": True,
            "has_workflow": True,
            "translated_template_mappings": translated_template_mappings or {}
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error getting translated template mappings for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translated template mappings: {exc}",
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
    
@router.post("/{conversation_id}/translate-all-fields", response_model=TranslationResponse)
async def translate_all_workflow_fields(
    conversation_id: UUID4,
    request: TranslateAllFieldsRequest,
    current_user: User = Depends(get_current_user),
) -> TranslationResponse:
    """
    Translate all field values in a workflow that haven't been manually edited or confirmed.
    Only translates fields where translated_status is not 'edited' or 'confirmed',
    unless force_retranslate is True.
    """
    try:
        # ── membership / auth ───────────────────────────────────────────────
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load existing workflow ──────────────────────────────────────────
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No workflow found for this conversation"
            )

        current_fields = workflow.fields or {}
        if not current_fields:
            return TranslationResponse(
                success=True,
                message="No fields found to translate",
                translated_fields={},
                skipped_fields={},
                errors={}
            )
        
        # ── determine source language if not provided ───────────────────────
        source_language = request.source_language
        if not source_language and workflow.translate_from:
            source_language = workflow.translate_from
        
        # ── process fields for translation ──────────────────────────────────
        fields_to_translate = {}
        skipped_fields = {}
        
        for field_key, field_metadata in current_fields.items():
            # Extract the actual value
            if hasattr(field_metadata, 'value'):
                field_value = field_metadata.value
                translated_status = getattr(field_metadata, 'translated_status', 'pending')
            elif isinstance(field_metadata, dict):
                field_value = field_metadata.get('value')
                translated_status = field_metadata.get('translated_status', 'pending')
            else:
                field_value = field_metadata
                translated_status = 'pending'
            
            # Skip if no value to translate
            if not field_value or (isinstance(field_value, str) and not field_value.strip()):
                skipped_fields[field_key] = "No value to translate"
                continue
            
            # Skip if already edited/confirmed and not forcing retranslation
            if not request.force_retranslate and translated_status in ['edited', 'confirmed']:
                skipped_fields[field_key] = f"Status is '{translated_status}' - skipping translation"
                continue
            
            # Add to translation queue
            if isinstance(field_value, str):
                fields_to_translate[field_key] = field_value
            else:
                # Convert non-string values to string for translation
                fields_to_translate[field_key] = str(field_value)
        
        if not fields_to_translate:
            return TranslationResponse(
                success=True,
                message="No fields eligible for translation",
                translated_fields={},
                skipped_fields=skipped_fields,
                errors={}
            )
        
        # ── perform translations ────────────────────────────────────────────
        try:
            translations = translation_service.translate_multiple_fields(
                fields_to_translate,
                request.target_language,
                source_language,
                request.use_gemini
            )
        except Exception as translation_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation service error: {str(translation_error)}"
            )
        
        # ── update workflow fields with translations ────────────────────────
        updated_fields = {}
        translation_errors = {}
        
        for field_key, field_metadata in current_fields.items():
            if field_key in translations:
                # Update the field with translation
                if hasattr(field_metadata, 'value'):
                    # FieldMetadata object
                    updated_field = {
                        "value": field_metadata.value,
                        "value_status": field_metadata.value_status,
                        "translated_value": translations[field_key],
                        "translated_status": "translated"
                    }
                elif isinstance(field_metadata, dict):
                    # Dictionary format
                    updated_field = {
                        "value": field_metadata.get('value'),
                        "value_status": field_metadata.get('value_status', 'pending'),
                        "translated_value": translations[field_key],
                        "translated_status": "translated"
                    }
                else:
                    # Simple value format
                    updated_field = {
                        "value": field_metadata,
                        "value_status": "pending",
                        "translated_value": translations[field_key],
                        "translated_status": "translated"
                    }
                
                updated_fields[field_key] = updated_field
            else:
                # Keep existing field unchanged
                if hasattr(field_metadata, 'value'):
                    updated_fields[field_key] = {
                        "value": field_metadata.value,
                        "value_status": field_metadata.value_status,
                        "translated_value": field_metadata.translated_value,
                        "translated_status": field_metadata.translated_status
                    }
                elif isinstance(field_metadata, dict):
                    updated_fields[field_key] = field_metadata
                else:
                    updated_fields[field_key] = {
                        "value": field_metadata,
                        "value_status": "pending",
                        "translated_value": None,
                        "translated_status": "pending"
                    }
        
        # ── update database ─────────────────────────────────────────────────
        update_data = {
            "fields": json.dumps(updated_fields),
            "updated_at": "now()"
        }
        
        result = supabase_client.client.table("workflows").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow with translations"
            )

        # ── update agent_state table ────────────────────────────────────────
        try:
            agent_state_result = supabase_client.client.table("agent_state").select("state_data").eq(
                "conversation_id", str(conversation_id)
            ).execute()
            
            if agent_state_result.data:
                current_state_data = agent_state_result.data[0].get("state_data", {})
                
                if "current_document_in_workflow_state" not in current_state_data:
                    current_state_data["current_document_in_workflow_state"] = {}
                
                current_state_data["current_document_in_workflow_state"]["fields"] = updated_fields
                
                supabase_client.client.table("agent_state").update({
                    "state_data": current_state_data,
                    "updated_at": "now()"
                }).eq("conversation_id", str(conversation_id)).execute()
        except Exception as agent_state_error:
            print(f"Warning: Failed to update agent_state: {agent_state_error}")
        
        # ── prepare response ────────────────────────────────────────────────
        translated_fields_response = {k: v for k, v in translations.items()}
        
        return TranslationResponse(
            success=True,
            message=f"Successfully translated {len(translated_fields_response)} fields",
            translated_fields=translated_fields_response,
            skipped_fields=skipped_fields,
            errors=translation_errors
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error translating all workflow fields for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to translate workflow fields: {exc}",
        )
    

@router.post("/{conversation_id}/translate-field", response_model=Dict[str, Any])
async def translate_single_workflow_field(
    conversation_id: UUID4,
    request: TranslateSingleFieldRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Translate a specific field value regardless of its current status.
    This will update the translated_value and set translated_status to 'translated'.
    """
    try:
        # ── membership / auth ───────────────────────────────────────────────
        _ = _guard_membership(str(conversation_id), current_user)

        # ── load existing workflow ──────────────────────────────────────────
        workflow = load_workflow_by_conversation(supabase_client, str(conversation_id))
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No workflow found for this conversation"
            )

        current_fields = workflow.fields or {}
        
        if request.field_key not in current_fields:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Field '{request.field_key}' not found in workflow"
            )
        
        # ── get field value ─────────────────────────────────────────────────
        field_metadata = current_fields[request.field_key]
        
        if hasattr(field_metadata, 'value'):
            field_value = field_metadata.value
        elif isinstance(field_metadata, dict):
            field_value = field_metadata.get('value')
        else:
            field_value = field_metadata
        
        if not field_value or (isinstance(field_value, str) and not field_value.strip()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Field '{request.field_key}' has no value to translate"
            )
        
        # ── determine source language ───────────────────────────────────────
        source_language = request.source_language
        if not source_language and workflow.translate_from:
            source_language = workflow.translate_from
        
        # ── perform translation ─────────────────────────────────────────────
        try:
            translated_value = translation_service.translate_field_value(
                str(field_value),
                request.target_language,
                source_language,
                field_context=request.field_key,
                use_gemini=request.use_gemini
            )
        except Exception as translation_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation service error: {str(translation_error)}"
            )
        
        # ── update field with translation ───────────────────────────────────
        if hasattr(field_metadata, 'value'):
            updated_field = {
                "value": field_metadata.value,
                "value_status": field_metadata.value_status,
                "translated_value": translated_value,
                "translated_status": "translated"
            }
        elif isinstance(field_metadata, dict):
            updated_field = {
                "value": field_metadata.get('value'),
                "value_status": field_metadata.get('value_status', 'pending'),
                "translated_value": translated_value,
                "translated_status": "translated"
            }
        else:
            updated_field = {
                "value": field_metadata,
                "value_status": "pending",
                "translated_value": translated_value,
                "translated_status": "translated"
            }
        
        # ── update workflow fields ──────────────────────────────────────────
        updated_fields = {}
        for field_key, field_data in current_fields.items():
            if field_key == request.field_key:
                updated_fields[field_key] = updated_field
            else:
                # Keep other fields unchanged
                if hasattr(field_data, 'value'):
                    updated_fields[field_key] = {
                        "value": field_data.value,
                        "value_status": field_data.value_status,
                        "translated_value": field_data.translated_value,
                        "translated_status": field_data.translated_status
                    }
                elif isinstance(field_data, dict):
                    updated_fields[field_key] = field_data
                else:
                    updated_fields[field_key] = {
                        "value": field_data,
                        "value_status": "pending",
                        "translated_value": None,
                        "translated_status": "pending"
                    }
        
        # ── update database ─────────────────────────────────────────────────
        update_data = {
            "fields": json.dumps(updated_fields),
            "updated_at": "now()"
        }
        
        result = supabase_client.client.table("workflows").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update workflow with translation"
            )

        # ── update agent_state table ────────────────────────────────────────
        try:
            agent_state_result = supabase_client.client.table("agent_state").select("state_data").eq(
                "conversation_id", str(conversation_id)
            ).execute()
            
            if agent_state_result.data:
                current_state_data = agent_state_result.data[0].get("state_data", {})
                
                if "current_document_in_workflow_state" not in current_state_data:
                    current_state_data["current_document_in_workflow_state"] = {}
                
                if "fields" not in current_state_data["current_document_in_workflow_state"]:
                    current_state_data["current_document_in_workflow_state"]["fields"] = {}
                
                current_state_data["current_document_in_workflow_state"]["fields"][request.field_key] = updated_field
                
                supabase_client.client.table("agent_state").update({
                    "state_data": current_state_data,
                    "updated_at": "now()"
                }).eq("conversation_id", str(conversation_id)).execute()
        except Exception as agent_state_error:
            print(f"Warning: Failed to update agent_state: {agent_state_error}")

        return {
            "success": True,
            "message": f"Successfully translated field '{request.field_key}'",
            "field_key": request.field_key,
            "original_value": field_value,
            "translated_value": translated_value,
            "source_language": source_language,
            "target_language": request.target_language,
            "translation_method": "gemini" if request.use_gemini else "google_translate",
            "updated_field": updated_field
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error translating single workflow field for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to translate workflow field: {exc}",
        )
    

    # ======================== NEW MODELS AND HELPERS FOR /insert-text-enhanced ========================
class FieldMetadata(BaseModel):
    value: Optional[str] = None
    value_status: str = "pending"
    translated_value: Optional[str] = None
    translated_status: str = "pending"

class FillTextEnhancedRequest(BaseModel):
    template_id: str
    template_translated_id: str
    isTranslated: bool
    fields: Dict[str, FieldMetadata]

def get_local_unicode_font():
    """
    Get the path to the local NotoSans font that supports Greek characters.
    Returns the font file path or None if font is not found.
    """
    try:
        # Path to your local NotoSans font
        font_path = os.path.join("fonts", "NotoSans-Regular.ttf")
        
        # Check if font exists
        if os.path.exists(font_path):
            print(f"Found local Unicode font at: {font_path}")
            return font_path
        else:
            print(f"Local font not found at: {font_path}")
            # Try alternative paths
            alternative_paths = [
                "./fonts/NotoSans-Regular.ttf",
                "fonts/NotoSans-Regular.ttf",
                os.path.abspath("fonts/NotoSans-Regular.ttf")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found local Unicode font at alternative path: {alt_path}")
                    return alt_path
            
            print("No local Unicode font found in any expected location")
            return None
            
    except Exception as e:
        print(f"Error accessing local Unicode font: {e}")
        return None

def has_unicode_text(text: str) -> bool:
    """
    Check if text contains Unicode characters that need special font handling.
    """
    return any(ord(char) > 127 for char in text)

def get_safe_font(requested_font: str, has_unicode: bool = False) -> str:
    """
    Map font names to PyMuPDF-compatible font names.
    For Unicode text, we'll use external font files.
    """
    if has_unicode:
        # For Unicode text, we'll return a flag to use external font
        return "unicode-font-needed"
    
    font_mapping = {
        "Helvetica": "helv",
        "Arial": "helv",
        "Times": "times-roman",
        "Times New Roman": "times-roman",
        "Courier": "cour",
        "Courier New": "cour",
    }
    
    # Return mapped font or default to Helvetica
    return font_mapping.get(requested_font, "helv")

# ======================== NEW ENDPOINT ========================
@router.post(
    "/generate/insert-text-enhanced",
    summary="Insert text values into PDF using template data with translation support"
)
async def insert_text_enhanced(
    request: FillTextEnhancedRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Insert text into PDF using template configuration from Supabase database with translation support.
    Downloads the PDF from the file_url stored in the template record.
    
    Args:
        request: Request containing:
            - template_id: UUID of the original template from Supabase
            - template_translated_id: UUID of the translated template from Supabase
            - isTranslated: Boolean flag to determine which template and values to use
            - fields: Dictionary of field mappings with FieldMetadata objects
    
    Returns:
        Modified PDF file
    """
    
    # 1️⃣ Determine which template to use based on isTranslated flag
    target_template_id = request.template_translated_id if request.isTranslated else request.template_id
    
    # 2️⃣ Fetch template from Supabase (get both info_json and file_url)
    try:
        response = supabase_client.client.table("templates").select("info_json, file_url").eq("id", target_template_id).execute()
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Template with ID {target_template_id} not found")
        
        template_info = response.data[0]["info_json"]
        file_url = response.data[0]["file_url"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
        if not file_url:
            raise HTTPException(status_code=400, detail="Template does not have a file_url")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # 3️⃣ Download PDF from file_url
    try:
        async with httpx.AsyncClient() as client:
            pdf_response = await client.get(file_url)
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF from file_url: {str(e)}")
    
    # 4️⃣ Get local Unicode font for Greek text support
    unicode_font_path = get_local_unicode_font()
    if not unicode_font_path:
        print("Warning: No local Unicode font found. Greek text may not display correctly.")
    
    # 5️⃣ Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Get the field key
        field_key = field.get("key", "")
        
        # Check if we have this field in our request
        if field_key in request.fields:
            field_metadata = request.fields[field_key]
            
            # Determine which value to use based on isTranslated flag
            if request.isTranslated:
                # Use translated_value if available
                if field_metadata.translated_value is not None:
                    updated_field["value"] = field_metadata.translated_value
            else:
                # Use regular value if available
                if field_metadata.value is not None:
                    updated_field["value"] = field_metadata.value
        
        updated_fields.append(updated_field)
    
    # 6️⃣ Open PDF document
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")
    
    # 7️⃣ Insert text for each field that has a value
    try:
        for field in updated_fields:
            # Skip fields without values
            if not field.get("value"):
                continue
                
            page = doc[field["page_number"] - 1]
            position = field["position"]
            bbox = Rect(position["x0"], position["y0"], position["x1"], position["y1"])
            
            # Font handling with robust fallback for Unicode/Greek text
            font_info = field["font"]
            requested_font = font_info.get("name", "Helvetica")
            text_value = field["value"]
            
            # Check if text contains Unicode characters
            has_unicode_chars = has_unicode_text(text_value)
            
            # FORCE TEXT COLOR TO ALWAYS BE BLACK - Override any template color
            r, g, b = 0, 0, 0  # Black color (RGB: 0, 0, 0)
            
            # Alignment handling
            alignment = field.get("alignment", "left").lower()
            font_size = font_info.get("size", 10)
            
            # Try to insert text with proper Unicode support
            text_inserted = False
            
            # Method 1: Try with local NotoSans font if available and text has Unicode chars
            if unicode_font_path and has_unicode_chars:
                try:
                    # Load the local NotoSans font
                    with open(unicode_font_path, "rb") as font_file:
                        fontfile = font_file.read()
                    
                    font = fitz.Font(fontbuffer=fontfile)
                    
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        # Use the font object directly with insert_text
                        text_writer = fitz.TextWriter(page.rect, color=(r, g, b))  # Set color here
                        text_writer.append(
                            (position["x0"], baseline),
                            text_value,
                            font=font,
                            fontsize=font_size
                        )
                        text_writer.write_text(page)
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        # For textbox, we need to use the fontname from the font
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=font.name,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted Unicode text with NotoSans font (black): {text_value[:50]}...")
                    
                except Exception as font_error:
                    print(f"NotoSans font insertion failed: {font_error}")
            
            # Method 2: Try with local font for regular text (even if not Unicode)
            elif unicode_font_path and not has_unicode_chars:
                try:
                    # Use NotoSans for all text for consistency
                    with open(unicode_font_path, "rb") as font_file:
                        fontfile = font_file.read()
                    
                    font = fitz.Font(fontbuffer=fontfile)
                    
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        # Use the font object directly with insert_text
                        text_writer = fitz.TextWriter(page.rect, color=(r, g, b))  # Set color here
                        text_writer.append(
                            (position["x0"], baseline),
                            text_value,
                            font=font,
                            fontsize=font_size
                        )
                        text_writer.write_text(page)
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        # For textbox, we need to use the fontname from the font
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=font.name,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted text with NotoSans font (black): {text_value[:50]}...")
                    
                except Exception as font_error:
                    print(f"NotoSans font insertion failed for regular text: {font_error}")
            
            # Method 3: Fallback to built-in fonts with encoding
            if not text_inserted:
                safe_font = get_safe_font(requested_font, has_unicode_chars)
                if safe_font == "unicode-font-needed":
                    safe_font = "helv"  # Fallback to built-in
                
                try:
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        page.insert_text(
                            (position["x0"], baseline),
                            text_value,
                            fontname=safe_font,
                            fontsize=font_size,
                            color=(r, g, b)  # Always black
                        )
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=safe_font,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted text with built-in font (black): {text_value[:50]}...")
                    
                except Exception as builtin_error:
                    print(f"Built-in font insertion failed: {builtin_error}")
            
            # Method 4: Last resort - try without encoding
            if not text_inserted:
                try:
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        page.insert_text(
                            (position["x0"], baseline),
                            text_value,
                            fontname="helv",
                            fontsize=font_size,
                            color=(r, g, b)  # Always black
                        )
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname="helv",
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    print(f"Inserted text with last resort method (black): {text_value[:50]}...")
                    
                except Exception as final_error:
                    print(f"All text insertion methods failed for: {text_value[:50]}... Error: {final_error}")
        
        # 8️⃣ Save modified PDF to buffer
        buf = io.BytesIO()
        try:
            # Save with garbage collection and compression for better Unicode handling
            doc.save(buf, garbage=4, deflate=True, clean=True)
            print(f"PDF saved successfully with advanced options")
        except Exception as save_error:
            print(f"Error saving with advanced options: {save_error}")
            # Fallback to basic save
            buf = io.BytesIO()  # Reset buffer
            doc.save(buf)
            print(f"PDF saved with basic options")
        
        # Always close the document
        doc.close()
        doc = None  # Clear reference
        
        # Get final buffer size AFTER all operations are complete
        final_buffer_size = buf.tell()
        buf.seek(0)  # Reset position for reading
        
        # Validate PDF content
        if final_buffer_size == 0:
            raise HTTPException(status_code=500, detail="Generated PDF is empty")
            
        # Check if buffer contains PDF header
        pdf_header = buf.read(4)
        buf.seek(0)  # Reset position again
        print(f"PDF header: {pdf_header}")
        
        if not pdf_header.startswith(b'%PDF'):
            raise HTTPException(status_code=500, detail="Generated file is not a valid PDF")
        
    except Exception as e:
        # Make sure to close the document even if there's an error
        if doc:
            doc.close()
        raise HTTPException(status_code=500, detail=f"PDF modification failed: {str(e)}")
    
    # 9️⃣ Return the modified PDF using StreamingResponse
    template_type = "translated" if request.isTranslated else "original"
    
    # Debug: Log response details
    print(f"Returning PDF: size={final_buffer_size} bytes, template_type={template_type}")
    
    headers = {
        "Content-Disposition": f'attachment; filename="filled_{template_type}_template_{target_template_id}.pdf"',
        "Cache-Control": "no-cache"
    }
    
    print(f"Response headers: {headers}")
    
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),  # Create a fresh BytesIO with the complete data
        media_type="application/pdf", 
        headers=headers
    )

    