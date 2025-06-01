# backend/src/agent/agent_state.py
from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(BaseModel):
    """
    Enhanced state for document-translation workflow with comprehensive field tracking.
    Using Pydantic BaseModel for better serialization and validation.
    """
    
    # ── Chat history & context ─────────────────────────────────────────────
    # All chat turns (HumanMessage, AIMessage, ToolMessage, etc.)
    messages: List[BaseMessage] = Field(default_factory=list)
    # A linearized record of past messages used for LLM context
    conversation_history: List[BaseMessage] = Field(default_factory=list)
    
    # Unique identifiers
    conversation_id: str = Field(default="")
    user_id: str = Field(default="")
    
    # The overall status of the current translation workflow
    workflow_status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    
    # Any additional context flags or small data you want to carry around
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # ── File upload & extraction ───────────────────────────────────────────
    # Supabase ID of the uploaded file (PDF, image, Word doc, etc.)
    user_upload_id: str = Field(default="")
    
    # Metadata about the uploaded file, e.g. { "name": "...", "size": 12345, "mime_type": "application/pdf" }
    user_upload: Dict[str, Any] = Field(default_factory=dict)
    
    # The fields that OCR + LLM have extracted so far, e.g. { "first_name": "David", "last_name": "Caparro", ... }
    extracted_required_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Which fields have been successfully filled (either by OCR or by user correction)
    filled_required_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # The same keys/values, but translated into the target language
    translated_required_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Which fields remain missing or unknown, e.g. { "birth_date": None, "address": None }
    missing_required_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # ── Template & translation info ────────────────────────────────────────
    # "English", "Greek", "Japanese", etc.
    translate_from: str = Field(default="")
    translate_to: str = Field(default="")
    
    # Supabase ID of the JSON template we'll use (once we know which one matches)
    template_id: str = Field(default="")
    
    # A mapping of placeholder keys to human-readable field names,
    # e.g. { "first_name": "First Name", "last_name": "Last Name", ... }
    template_required_fields: Dict[str, str] = Field(default_factory=dict)
    
    # ── Document output & versioning ───────────────────────────────────────
    # After fill_template runs, we store the Supabase record ID of the newly generated PDF/Word file
    document_version_id: str = Field(default="")
    
    # ── Workflow control & audit ───────────────────────────────────────────
    # A simple log of which node names have already completed,
    # e.g. ["load_state_graph", "analyze_doc", "translate_required_fields"]
    steps_done: List[str] = Field(default_factory=list)
    
    class Config:
        # Allow arbitrary types (needed for BaseMessage objects)
        arbitrary_types_allowed = True
        # Use enum values for serialization but keep enum types in memory
        use_enum_values = False  # Changed to False to keep enum objects
        # Custom serializer for enums
        json_encoders = {
            WorkflowStatus: lambda v: v.value if hasattr(v, 'value') else str(v)
        }