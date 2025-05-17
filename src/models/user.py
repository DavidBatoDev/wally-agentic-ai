"""
User models for authentication and profile information.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID

class User(BaseModel):
    """Base user model with authentication details."""
    id: str
    email: Optional[str] = None

class UserProfile(BaseModel):
    """Extended user profile information."""
    id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserProfileCreate(BaseModel):
    """Schema for creating or updating a user profile."""
    full_name: str
    avatar_url: Optional[str] = None

class UserResponse(BaseModel):
    """Response model for user information."""
    id: str
    email: Optional[str] = None
    profile: Optional[UserProfile] = None