# backend/src/config.py
"""
Configuration module for the agentic AI system with improved JWT handling.
"""

import os
from dotenv import load_dotenv
from functools import lru_cache
import base64
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

class Settings:
    """Application settings."""
    
    # App settings
    APP_NAME: str = "Wally-Chat API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Supabase settings
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "my-supabase-url")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "my-supabase-key")
    
    # Raw JWT secret from environment variable
    _RAW_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "my-jwt-secret")
    
    # Decoded JWT secret (processed during initialization)
    SUPABASE_JWT_SECRET: str = None
    
    # Service role key
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Authentication settings
    VERIFY_USER_EXISTS: bool = os.getenv("VERIFY_USER_EXISTS", "False").lower() in ("true", "1", "t")

    DEBUG: bool = True
    
    # Agent settings
    MAX_AGENT_ITERATIONS: int = 5
    DEFAULT_AGENT_TEMPERATURE: float = 0.2

    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",  # Next.js frontend default port
        "http://localhost:8000",  # FastAPI backend
        # Add other allowed origins if needed
    ]

    # LLM settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "my-gemini-api-key")
    
    # Storage settings
    STORAGE_BUCKET_NAME: str = "user-uploads"
    
    # Security settings
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    def __init__(self):
        """Initialize settings and process JWT secret."""
        # Process the JWT secret
        self._process_jwt_secret()
        
        # Log debug information
        # if self.DEBUG:
        #     logger.info(f"App name: {self.APP_NAME}")
        #     logger.info(f"Supabase URL: {self.SUPABASE_URL}")
        #     logger.info(f"JWT secret length: {len(self.SUPABASE_JWT_SECRET) if self.SUPABASE_JWT_SECRET else 0}")
    
    def _process_jwt_secret(self):
        """Process the JWT secret for proper format."""
        # Store the raw secret as fallback
        self.SUPABASE_JWT_SECRET = self._RAW_JWT_SECRET
        
        # Try to fix common JWT secret issues
        try:
            # If the secret contains URL-unsafe characters, it might be base64 encoded
            if '/' in self._RAW_JWT_SECRET or '+' in self._RAW_JWT_SECRET:
                # Add padding if necessary
                padded_secret = self._RAW_JWT_SECRET
                missing_padding = len(self._RAW_JWT_SECRET) % 4
                if missing_padding:
                    padded_secret += '=' * (4 - missing_padding)
                
                # Try to decode it (just to test if it's valid base64)
                base64.b64decode(padded_secret)
                
                # If we got here, it's valid base64, so let's keep the padded version
                self.SUPABASE_JWT_SECRET = padded_secret
                logger.info("JWT secret appears to be base64 encoded, padded if needed")
        except Exception as e:
            # If there's an error, keep the original
            logger.warning(f"Error processing JWT secret: {str(e)}")
            logger.warning("Using raw JWT secret as fallback")


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()