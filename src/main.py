# backend/main.py
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

# Set up proper logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Use absolute imports instead of relative imports
from src.config import get_settings
from src.routers import agent, upload, user

# Load settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(user.router, prefix="/api/user", tags=["User"])

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Log startup information
# @app.on_event("startup")
# async def startup_event():
#     logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
#     logger.info(f"Debug mode: {settings.DEBUG}")
#     logger.info(f"CORS origins: {settings.CORS_ORIGINS}")
    
#     # Verify JWT secret is configured
#     if settings.SUPABASE_JWT_SECRET:
#         logger.info(f"JWT secret is configured (length: {len(settings.SUPABASE_JWT_SECRET)})")
#     else:
#         logger.warning("JWT secret is not configured!")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)