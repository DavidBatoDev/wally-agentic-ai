# backend/src/models/conversation.py
"""
Models for conversation, messages and agents.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from datetime import datetime


class Conversation(BaseModel):
    """Schema for a conversation."""
    id: UUID
    profile_id: UUID
    title: Optional[str] = None
    summary: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    title: Optional[str] = None


class Message(BaseModel):
    """Schema for a message in a conversation."""
    id: UUID
    conversation_id: UUID
    sender: Literal[
        "user",
        "assistant",
        "system",   
        "model",
        "tools"  
    ]
    kind: Literal[
        "text",
        "file",
        "action",
        "file_card",
        "buttons",
        "inputs"
    ]
    body: str
    created_at: datetime

    class Config:
        from_attributes = True


class TextMessageCreate(BaseModel):
    """Schema for creating a new text message."""
    conversation_id: UUID
    body: str


class ActionMessageCreate(BaseModel):
    """Schema for creating a new action message."""
    conversation_id: UUID
    action: str
    values: Optional[Dict[str, Any]] = None


class ButtonConfig(BaseModel):
    """Schema for a button configuration."""
    label: str
    action: str


class InputConfig(BaseModel):
    """Schema for an input field configuration."""
    key: str
    label: str
    value: Optional[str] = None


class ButtonsMessage(BaseModel):
    """Schema for a message with buttons."""
    prompt: str
    buttons: List[ButtonConfig]


class InputsMessage(BaseModel):
    """Schema for a message with input fields."""
    prompt: str
    inputs: List[InputConfig]


class AgentMemory(BaseModel):
    """Schema for agent memory/context."""
    conversation_id: UUID
    content: str
    embedding: List[float]
    meta: Optional[Dict[str, Any]] = None