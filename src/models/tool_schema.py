# backend/src/models/tool_schema.py
"""
Tool definitions and schemas for LLM agent.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal


class ToolInput(BaseModel):
    """Base schema for tool inputs."""
    pass


class ToolOutput(BaseModel):
    """Base schema for tool outputs."""
    pass


class OCRToolInput(ToolInput):
    """Input for OCR tool."""
    file_id: str = Field(..., description="The ID of the file to perform OCR on")
    

class OCRToolOutput(ToolOutput):
    """Output from OCR tool."""
    text: str = Field(..., description="The extracted text from the image")
    confidence: float = Field(..., description="Confidence score of the OCR result")


class TranslateToolInput(ToolInput):
    """Input for translation tool."""
    text: str = Field(..., description="The text to translate")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
    target_language: str = Field(..., description="Target language code")


class TranslateToolOutput(ToolOutput):
    """Output from translation tool."""
    translated_text: str = Field(..., description="The translated text")
    detected_source_language: str = Field(..., description="Detected source language code")


class ExtractFieldsToolInput(ToolInput):
    """Input for field extraction tool."""
    text: str = Field(..., description="Text to extract fields from")
    template_id: Optional[str] = Field(None, description="Template ID to use for extraction")
    doc_type: Optional[str] = Field(None, description="Document type if template_id is not provided")


class ExtractFieldsToolOutput(ToolOutput):
    """Output from field extraction tool."""
    fields: Dict[str, Any] = Field(..., description="Dictionary of extracted fields")
    missing_fields: List[str] = Field(default_factory=list, description="List of fields that could not be extracted")


class SearchMemoryToolInput(ToolInput):
    """Input for memory search tool."""
    conversation_id: str = Field(..., description="Conversation ID to search within")
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, description="Maximum number of results to return")


class SearchMemoryToolOutput(ToolOutput):
    """Output from memory search tool."""
    results: List[Dict[str, Any]] = Field(..., description="List of matching memory entries")
    

class ToolDescription(BaseModel):
    """Description of a tool available to the agent."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class AgentAction(BaseModel):
    """Action selected by the agent."""
    tool: str = Field(..., description="Name of the tool to use")
    tool_input: Dict[str, Any] = Field(..., description="Input parameters for the tool")
    thought: str = Field(..., description="Agent's reasoning for selecting this tool")