# backend/src/routers/workflow.py
"""
Workflow API router for managing workflow states and operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, UUID4
import json

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
    
    