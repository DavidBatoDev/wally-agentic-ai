# backend/src/dependencies/agent.py
"""
Dependencies for the agent orchestrator.
"""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.orchestrator import AgentOrchestrator
from src.agent.setup_orchestrator import setup_agent_orchestrator
from src.db.db_client import supabase_client
from src.config import get_settings

settings = get_settings()

@lru_cache()
def get_agent_orchestrator() -> AgentOrchestrator:
    """
    Get or create the agent orchestrator singleton.
    Uses lru_cache to ensure only one instance is created.
    
    Returns:
        The agent orchestrator instance
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.DEFAULT_AGENT_TEMPERATURE,
        streaming=True
    )
    
    # Set up the agent orchestrator
    agent_orchestrator = setup_agent_orchestrator(
        llm=llm, 
        supabase_client=supabase_client
    )
    
    return agent_orchestrator