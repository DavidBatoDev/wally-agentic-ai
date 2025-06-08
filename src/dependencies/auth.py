# backend/src/dependencies/auth.py
"""
Enhanced authentication for FastAPI routes using Supabase JWT verification.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt.exceptions import PyJWTError
from typing import Optional
import requests
import logging

from src.config import get_settings

# Initialize settings
settings = get_settings()
from src.models.user import User

# Set up logging
# logger = logging.getLogger(__name__)

# Use standard HTTP Bearer for auth
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Verify the JWT token using Supabase JWT secret and return the current user.
    
    Args:
        credentials: The HTTP Authorization header containing the JWT
        
    Returns:
        User: A User object with the decoded user information
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Get the token
        token = credentials.credentials
        # logger.info(f"Token received: {token[:10]}...")
        
        # Prepare JWT options to disable audience validation since we know this causes issues
        jwt_options = {
            'verify_signature': True,
            'verify_aud': False,  # Disable audience validation
            'verify_iat': True,
            'verify_exp': True,
            'verify_nbf': False,
            'verify_iss': False,  # Don't verify issuer
            'verify_sub': False,
            'verify_jti': False,
            'verify_at_hash': False,
            'require_aud': False,
            'require_iat': False,
            'require_exp': False,
            'require_nbf': False,
            'require_iss': False,
            'require_sub': False,
            'require_jti': False,
            'require_at_hash': False,
            'leeway': 0,
        }
        
        # Get algorithm from token header
        token_header = jwt.get_unverified_header(token)
        alg = token_header.get('alg', 'HS256')
        
        # Use the raw JWT secret directly - debug output showed this works!
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[alg],
            options=jwt_options,
        )
        
        # Extract user ID from the token - 'sub' is the standard claim
        user_id = payload.get("sub")
        if user_id is None:
            # Fallbacks if 'sub' doesn't exist
            user_id = payload.get("user_id") or payload.get("id")
            if user_id is None:
                # logger.error("No user ID found in token payload")
                raise credentials_exception
        
        # Extract email from the token
        email = payload.get("email")
        
        # logger.info(f"Successfully authenticated user: {user_id}, email: {email}")
        
        return User(
            id=user_id,
            email=email,
        )
    except PyJWTError as e:
        # logger.error(f"JWT verification error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        # logger.error(f"Unexpected error in authentication: {str(e)}")
        raise credentials_exception

# Optional dependency for routes that need the current user but can work without authentication
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Try to get the current user, but don't raise an exception if authentication fails.
    
    Args:
        credentials: The HTTP Authorization header containing the JWT
        
    Returns:
        Optional[User]: A User object if authentication succeeds, None otherwise
    """
    try:
        if credentials:
            return await get_current_user(credentials)
        return None
    except HTTPException:
        return None