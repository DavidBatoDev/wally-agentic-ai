# backend/src/main.py
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
from src.routers import conversations, messages, user, uploads, workflow
from src.dependencies.agent import get_langgraph_orchestrator

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
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])
app.include_router(messages.router, prefix="/api/messages", tags=["Messages"])
app.include_router(uploads.router, prefix="/api/uploads", tags=["Upload"])
app.include_router(user.router, prefix="/api/user", tags=["User"])
app.include_router(workflow.router, prefix="/api/workflow", tags=["Workflow"])

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
    # Initialize the agent_orchestrator to ensure it's ready when needed
    # This is optional but ensures the LLM is initialized at startup
    get_langgraph_orchestrator()
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)