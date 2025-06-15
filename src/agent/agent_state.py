# backend/src/agent/agent_state.py

from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.schema import BaseMessage
from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FieldMetadata(BaseModel):
    value: Any                                   # Main field value (extracted or filled)
    value_status: str = "pending"                # Status for value: "pending", "ocr", "edited", or "confirmed"

    translated_value: Optional[str] = None       # LLM translation of the value (if available)
    translated_status: str = "pending"           # Status for translated_value: "pending", "translated", "edited", or "confirmed"


class Upload(BaseModel):
    """
    Represents a single file_upload. Stores metadata and (eventual) analysis results.
    """
    file_id: str = ""                # e.g. Supabase file ID
    mime_type: Optional[str] = ""    # e.g. "image/png/pdf"
    public_url: str = ""             # e.g. publicly accessible URL
    filename: Optional[str] = ""     # original filename
    size_bytes: Optional[int] = 0
    analysis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    is_templatable: Optional[bool] = False  # whether this upload can be used in templates
    translated_from: Optional[str] = 'English'  # Language this file was translated from, if applicable
    class Config:
        arbitrary_types_allowed = True

class CurrentDocumentInWorkflow(BaseModel):
    """
    Represents the current state of the document being processed.
    This includes the document's content, metadata, and any additional information
    needed for processing.
    """
    file_id: str = ""  # e.g. Supabase file ID (Will get the CurrentDocumentInWorkflow if a find_template_node is found)
    base_file_public_url: str = ""  # publicly accessible URL of the document (Will get the CurrentDocumentInWorkflow if a find_template_node is found)
    template_id: str = ""  # ID of the template used, if any (Will get this using the find_template_node)
    template_file_public_url: str = "" # template file url (Will get this using the find_template_node)
    template_translated_id: Optional[str] = ""  # ID of the translated template, if applicable (Will get this using the find_template_node)
    template_translated_file_public_url: Optional[str] = "" # template file url (Will get this using the find_template_node)
    template_required_fields: Dict[str, Any] = Field(default_factory=dict) # require (Will get this using the find_template_node)
    fields: Dict[str, FieldMetadata] = Field(default_factory=dict) # "{birth_date}": FieldMetadata(value="2023-01-01", value_status="confirmed")
    translate_to: Optional[str] = None  # Language to translate the document to, if applicable
    translate_from: Optional[str] = None  # Language the document is currently in, if applicable
    current_document_version_public_url: str = "" # ID of the current document version

    class Config:
        arbitrary_types_allowed = True

class AgentState(BaseModel):
    """
    Pydantic model holding the entire workflow state. Everything is JSON‐serializable,
    yet we keep rich `BaseMessage` objects in memory.
    """

    # 1) Chat history
    messages: List[BaseMessage] = Field(default_factory=list)
    conversation_history: List[BaseMessage] = Field(default_factory=list)
    # 2) Identifiers
    conversation_id: str = ""
    user_id: str = ""
    # 3) Status / context
    workflow_status: WorkflowStatus = WorkflowStatus.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)
    # 4) File uploads (split into two categories)
    user_uploads: List[Upload] = Field(default_factory=list)
    # 5) Detected user intent if the 
    translate_to: Optional[str] = ""  # we will get this node (ask_user_desired_language)
    # 7) Current Document We are Working On
    current_document_in_workflow_state: CurrentDocumentInWorkflow = Field(default_factory=CurrentDocumentInWorkflow) 
    # 8) Output & audit
    steps_done: List[str] = Field(default_factory=list)
    # 9) if WorkflowStatus.WAITING_CONFIRMATION go to this node
    current_pending_node: str = ""

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {WorkflowStatus: lambda v: v.value}

    @property
    def latest_upload(self) -> Optional[Upload]:
        """
        Return the most recent "normal" upload, or None if there are none.
        """
        return self.user_uploads[-1] if self.user_uploads else None