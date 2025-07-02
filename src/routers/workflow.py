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
from .language_helpers import normalize_language_code, validate_language_code, normalize_and_validate_language

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
    value: Optional[str] = None
    value_status: str = "manual"
    translated_value: Optional[str] = None
    translated_status: str = "pending"


class ReplaceWorkflowRequest(BaseModel):
    file_id: str
    base_file_public_url: Optional[str] = None
    template_id: str
    template_file_public_url: Optional[str] = None
    origin_template_mappings: Optional[Dict[str, Any]] = None
    fields: Optional[Dict[str, FieldMetadataDict]] = None
    template_translated_id: str
    template_translated_file_public_url: Optional[str] = None
    translated_template_mappings: Optional[Dict[str, Any]] = None
    translate_to: str
    translate_from: str
    shapes: Optional[List[Any]] = None
    deletion_rectangles: Optional[List[Any]] = None

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


def _parse_if_str(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val or {}


def _convert_array_to_object(val):
    """Convert array or string to object format expected by frontend."""
    if val is None:
        return {}
    
    # If it's a string, parse it first
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return {}
    
    # If it's already an object/dict, return it
    if isinstance(val, dict):
        return val
    
    # If it's an array, convert to object using ID as key
    if isinstance(val, list):
        result = {}
        for item in val:
            if isinstance(item, dict) and 'id' in item:
                result[item['id']] = item
        return result
    
    return {}


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
        "shapes": _convert_array_to_object(getattr(workflow_obj, 'shapes', None)),
        "deletion_rectangles": _convert_array_to_object(getattr(workflow_obj, 'deletion_rectangles', None)),
    }
    # Always parse mappings if they are strings
    if origin_template_mappings is not None:
        workflow_dict["origin_template_mappings"] = _parse_if_str(origin_template_mappings)
    if translated_template_mappings is not None:
        workflow_dict["translated_template_mappings"] = _parse_if_str(translated_template_mappings)
    # Always include info_json_custom, parsed as object if string
    if hasattr(workflow_obj, 'info_json_custom') and workflow_obj.info_json_custom is not None:
        val = workflow_obj.info_json_custom
        if isinstance(val, str):
            try:
                workflow_dict["info_json_custom"] = json.loads(val)
            except Exception:
                workflow_dict["info_json_custom"] = {}
        else:
            workflow_dict["info_json_custom"] = val
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
        
        # Update the specific field - preserve existing values for fields not provided
        existing_field = serialized_current_fields.get(request.field_key, {})
        updated_field = {
            "value": request.value if request.value is not None else existing_field.get("value"),
            "value_status": request.value_status if request.value is not None else existing_field.get("value_status", "pending"),
            "translated_value": request.translated_value if request.translated_value is not None else existing_field.get("translated_value"),
            "translated_status": request.translated_status if request.translated_value is not None else existing_field.get("translated_status", "pending")
        }
        
        serialized_current_fields[request.field_key] = updated_field
        
        # ── update workflow in database -----------------------------------------
        update_data = {
            "fields": serialized_current_fields,
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
            "fields": serialized_fields,
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
        
        # ── normalize and validate language codes ────────────────────────────
        target_language_code, target_is_valid = normalize_and_validate_language(request.target_language)
        
        if not target_is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported target language: '{request.target_language}'. Normalized to '{target_language_code}' but not supported by translation service."
            )
        
        source_language_code = None
        if source_language:
            source_language_code, source_is_valid = normalize_and_validate_language(source_language)
            if not source_is_valid:
                print(f"Warning: Unsupported source language '{source_language}', normalized to '{source_language_code}'. Translation will proceed with auto-detection.")
                # Don't raise error for source language - let the service auto-detect
                source_language_code = None
        
        print(f"Translation request - Target: '{request.target_language}' -> '{target_language_code}' (valid: {target_is_valid})")
        if source_language:
            print(f"Translation request - Source: '{source_language}' -> '{source_language_code}' (valid: {source_is_valid if source_language_code else 'auto-detect'})")
        
        # ── perform translations ────────────────────────────────────────────
        try:
            translations = translation_service.translate_multiple_fields(
                fields_to_translate,
                target_language_code,
                source_language_code,
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
                        "translated_value": getattr(field_metadata, 'translated_value', None),
                        "translated_status": getattr(field_metadata, 'translated_status', 'pending')
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
            "fields": updated_fields,
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
            message=f"Successfully translated {len(translated_fields_response)} fields from {source_language or 'auto-detected'} to {request.target_language}",
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
        source_language = workflow.translate_from
        
        # ── perform translation ─────────────────────────────────────────────
        try:
            print(field_value)
            translated_value = translation_service.translate_field_value(
                str(field_value),
                workflow.translate_to,
                source_language,
                field_context=request.field_key,
                use_gemini=request.use_gemini
            )
            print(translated_value)
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
            "fields": updated_fields,
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

@router.put("/{conversation_id}/replace", response_model=Dict[str, Any])
async def replace_entire_workflow(
    conversation_id: UUID4,
    request: ReplaceWorkflowRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Replace the entire workflow data for a conversation and upsert agent_state.
    This will completely replace the workflow row in the database with the provided data.
    It also upserts the agent_state table with filtered data (excluding shapes, deletion_rectangles, mappings).
    """
    try:
        # ── membership / auth ───────────────────────────────────────────────
        _ = _guard_membership(str(conversation_id), current_user)

        # ── serialize fields for database storage ───────────────────────────
        serialized_fields = {}
        if request.fields:
            serialized_fields = _serialize_fields_for_db(request.fields)

        # ── serialize mappings for database storage ──────────────────────── 
        origin_mappings = request.origin_template_mappings or {}
        translated_mappings = request.translated_template_mappings or {}

        # ── prepare workflow data for database ──────────────────────────────
        workflow_data = {
            "conversation_id": str(conversation_id),
            "file_id": request.file_id,
            "base_file_public_url": request.base_file_public_url,
            "template_id": request.template_id,
            "template_file_public_url": request.template_file_public_url,
            "origin_template_mappings": origin_mappings,
            "fields": serialized_fields,
            "template_translated_id": request.template_translated_id,
            "template_translated_file_public_url": request.template_translated_file_public_url,
            "translated_template_mappings": translated_mappings,
            "translate_to": request.translate_to,
            "translate_from": request.translate_from,
            "shapes": request.shapes,  # Store shapes as array
            "deletion_rectangles": request.deletion_rectangles,  # Store deletion_rectangles as array
            "updated_at": "now()"
        }

        # ── replace workflow in database (update or insert) ─────────────────
        # First try to update existing workflow
        update_data = {k: v for k, v in workflow_data.items() if k != "conversation_id"}
        update_data["updated_at"] = "now()"
        
        update_result = supabase_client.client.table("workflows").update(update_data).eq(
            "conversation_id", str(conversation_id)
        ).execute()

        if not update_result.data:
            # No existing workflow found, create a new one
            insert_result = supabase_client.client.table("workflows").insert(workflow_data).execute()
            
            if not insert_result.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create new workflow data"
                )
        # If update_result.data exists, the update was successful

        # ── prepare agent_state data (filtered) ─────────────────────────────
        # Exclude shapes, deletion_rectangles, origin_template_mappings, translated_template_mappings
        agent_state_workflow_data = {
            "file_id": request.file_id,
            "base_file_public_url": request.base_file_public_url,
            "template_id": request.template_id,
            "template_file_public_url": request.template_file_public_url,
            "fields": serialized_fields,
            "template_translated_id": request.template_translated_id,
            "template_translated_file_public_url": request.template_translated_file_public_url,
            "translate_to": request.translate_to,
            "translate_from": request.translate_from
        }

        # ── upsert agent_state table ────────────────────────────────────────
        try:
            # Get current agent_state or create new structure
            agent_state_result = supabase_client.client.table("agent_state").select("state_data").eq(
                "conversation_id", str(conversation_id)
            ).execute()

            if agent_state_result.data:
                # Update existing agent_state
                current_state_data = agent_state_result.data[0].get("state_data", {})
                current_state_data["current_document_in_workflow_state"] = agent_state_workflow_data

                agent_state_update_result = supabase_client.client.table("agent_state").update({
                    "state_data": current_state_data,
                    "updated_at": "now()"
                }).eq("conversation_id", str(conversation_id)).execute()

                if not agent_state_update_result.data:
                    print(f"Warning: Failed to update agent_state for conversation {conversation_id}")
            else:
                # Create new agent_state
                new_state_data = {
                    "current_document_in_workflow_state": agent_state_workflow_data
                }

                agent_state_insert_result = supabase_client.client.table("agent_state").insert({
                    "conversation_id": str(conversation_id),
                    "state_data": new_state_data,
                    "created_at": "now()",
                    "updated_at": "now()"
                }).execute()

                if not agent_state_insert_result.data:
                    print(f"Warning: Failed to create agent_state for conversation {conversation_id}")

        except Exception as agent_state_error:
            print(f"Error upserting agent_state for conversation {conversation_id}: {agent_state_error}")
            # Don't fail the entire request if agent_state upsert fails
            # The workflow table upsert was successful, so we can continue

        return {
            "success": True,
            "message": "Workflow data replaced successfully",
            "conversation_id": str(conversation_id),
            "workflow_data": {
                "file_id": request.file_id,
                "base_file_public_url": request.base_file_public_url,
                "template_id": request.template_id,
                "template_file_public_url": request.template_file_public_url,
                "origin_template_mappings": origin_mappings,
                "fields": serialized_fields,
                "template_translated_id": request.template_translated_id,
                "template_translated_file_public_url": request.template_translated_file_public_url,
                "translated_template_mappings": translated_mappings,
                "translate_to": request.translate_to,
                "translate_from": request.translate_from,
                "shapes": request.shapes,
                "deletion_rectangles": request.deletion_rectangles
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error replacing workflow for conversation {conversation_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to replace workflow: {exc}",
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
    template_mappings: Optional[Dict[str, Any]] = None

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
