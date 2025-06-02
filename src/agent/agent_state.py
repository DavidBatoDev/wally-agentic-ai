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

class UploadAnalysis(BaseModel):
    file_id: str
    mime_type: str
    doc_type: str
    variation: str = "standard"
    detected_language: str = "unknown"
    confidence: float = 0.0
    page_size: str | None = None
    page_count: int | None = None


class AgentState(BaseModel):
    """
    Pydantic model holding the entire workflow state.  Everything is JSON-serialisable
    and validated, yet keeps rich `BaseMessage` objects in memory.
    """

    # Chat history
    messages: List[BaseMessage] = Field(default_factory=list)
    conversation_history: List[BaseMessage] = Field(default_factory=list)

    # Identifiers
    conversation_id: str = ""
    user_id: str = ""

    # Status / context
    workflow_status: WorkflowStatus = WorkflowStatus.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)

    # File upload
    user_upload_id: str = ""
    user_upload_public_url: str = ""
    upload_analysis: UploadAnalysis | None = None

    # Field tracking
    extracted_required_fields: Dict[str, Any] = Field(default_factory=dict)
    filled_required_fields: Dict[str, Any] = Field(default_factory=dict)
    translated_required_fields: Dict[str, Any] = Field(default_factory=dict)
    missing_required_fields: Dict[str, Any] = Field(default_factory=dict)

    # Template / translation
    translate_from: Optional[str] = ""
    translate_to: Optional[str] = ""
    template_id: str = ""
    template_required_fields: Dict[str, str] = Field(default_factory=dict)

    # Output & audit
    document_version_id: str = ""
    steps_done: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {WorkflowStatus: lambda v: v.value}
