# backend/src/routers/user.py
"""
User API routes for profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Dict, Any
import jwt
import logging

from src.models.user import User, UserProfile, UserProfileCreate, UserResponse
from src.dependencies.auth import get_current_user, get_optional_user
from src.utils.supabase import supabase_client
from src.config import get_settings

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


@router.get("/debug-token")
async def debug_token(request: Request) -> Dict[str, Any]:
    """
    Debug endpoint to check token issues.
    
    Args:
        request: The request object to extract headers
        
    Returns:
        Dict: Token debug information
    """
    try:
        # Extract authorization header
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return {
                "status": "error",
                "message": "No Bearer token found in Authorization header",
                "headers_received": dict(request.headers)
            }
        
        # Extract token
        token = auth_header.replace("Bearer ", "")
        
        # Try to decode without verification
        try:
            debug_payload = jwt.decode(token, options={"verify_signature": False})
            
            # Check for common fields
            user_id = debug_payload.get("sub") or debug_payload.get("user_id") or debug_payload.get("id")
            email = debug_payload.get("email")
            
            return {
                "status": "debug_successful",
                "token_first_chars": token[:10] + "...",
                "decoded_payload": debug_payload,
                "found_user_id": user_id,
                "found_email": email,
                "jwt_secret_length": len(settings.SUPABASE_JWT_SECRET) if settings.SUPABASE_JWT_SECRET else 0
            }
        except Exception as e:
            return {
                "status": "decode_error",
                "error": str(e),
                "token_first_chars": token[:10] + "..." if token else "No token"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@router.get("/test-optional")
async def test_optional_auth(
    current_user: User = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Test endpoint that works with or without authentication.
    
    Args:
        current_user: The authenticated user (optional)
        
    Returns:
        Dict: Status message
    """
    if current_user:
        return {
            "authenticated": True,
            "user_id": current_user.id,
            "email": current_user.email
        }
    else:
        return {
            "authenticated": False,
            "message": "No valid authentication provided, but endpoint still works"
        }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the current user's profile information.
    
    Args:
        current_user: The authenticated user
        
    Returns:
        Dict: User information and profile
    """

    print(f"Current user: {current_user}")
    try:
        # Get user profile from Supabase
        user_profile = supabase_client.get_user_profile(user_id=current_user.id)
        print(f"User profile: {user_profile}")
        
        # Return user info with profile
        return {
            "id": current_user.id,
            "email": current_user.email,
            "profile": user_profile
        }
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}"
        )


@router.post("/profile", response_model=UserProfile)
async def create_or_update_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create or update the current user's profile.
    
    Args:
        profile_data: The profile data to update
        current_user: The authenticated user
        
    Returns:
        Dict: Updated profile information
    """
    try:
        # Check if profile exists
        existing_profile = await get_user_profile(current_user.id)
        
        profile_dict = profile_data.dict()
        profile_dict["user_id"] = current_user.id
        
        if existing_profile:
            # Update existing profile - use our custom client's methods
            response = supabase_client.client.table("profiles").update(profile_dict).eq("id", current_user.id).execute()
        else:
            # Create new profile - use our custom client's methods
            response = supabase_client.client.table("profiles").insert(profile_dict).execute()
        
        if response.data:
            # Add user_id and email to response
            profile_data = response.data[0]
            profile_data["id"] = current_user.id
            profile_data["email"] = current_user.email
            return profile_data
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )


async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Get a user's profile from Supabase.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        Dict: The user's profile information
    """
    try:
        # Fixed: Pass the user_id correctly to match supabase_client's expected parameter name
        response = supabase_client.get_user_profile(user_id=user_id)
        print(f"Response from Supabase: {response}")
        return response
    except Exception as e:
        raise Exception(f"Failed to get user profile: {str(e)}")