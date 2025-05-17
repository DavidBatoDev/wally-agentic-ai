# backend/src/routers/messages.py
"""
Router for user profile management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional

from ..dependencies.auth import get_current_user
from ..models.user import User, UserProfileCreate, UserResponse, UserProfile
from ..utils.db_client import supabase_client

router = APIRouter(
    prefix="/users",
    tags=["users"],
)


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Get the current user's profile.
    """
    try:
        # Get the user profile from Supabase
        profile = supabase_client.get_user_profile(current_user.id)
        
        # If profile doesn't exist yet, return just the basic user info
        if not profile:
            return UserResponse(
                id=current_user.id,
                email=current_user.email,
                profile=None,
            )
        
        # Convert to UserProfile model
        user_profile = UserProfile(
            id=profile.get("id"),
            email=current_user.email,
            full_name=profile.get("full_name"),
            avatar_url=profile.get("avatar_url"),
        )
        
        return UserResponse(
            id=current_user.id,
            email=current_user.email,
            profile=user_profile,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}",
        )


@router.post("/profile", response_model=UserProfile)
async def create_or_update_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_user),
) -> UserProfile:
    """
    Create or update the current user's profile.
    """
    try:
        # Check if profile exists
        existing_profile = supabase_client.get_user_profile(current_user.id)
        
        if existing_profile:
            # Update existing profile
            updated_data = {
                "full_name": profile_data.full_name,
            }
            
            if profile_data.avatar_url:
                updated_data["avatar_url"] = profile_data.avatar_url
            
            # Update in Supabase
            response = (
                supabase_client.client.table("profiles")
                .update(updated_data)
                .eq("id", current_user.id)
                .execute()
            )
            
            updated_profile = response.data[0] if response.data else None
            
            if not updated_profile:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update profile",
                )
            
            return UserProfile(
                id=updated_profile.get("id"),
                email=current_user.email,
                full_name=updated_profile.get("full_name"),
                avatar_url=updated_profile.get("avatar_url"),
            )
        else:
            # Create new profile
            new_profile = supabase_client.create_user_profile(
                user_id=current_user.id,
                full_name=profile_data.full_name,
                avatar_url=profile_data.avatar_url,
            )
            
            return UserProfile(
                id=new_profile.get("id"),
                email=current_user.email,
                full_name=new_profile.get("full_name"),
                avatar_url=new_profile.get("avatar_url"),
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create/update profile: {str(e)}",
        )