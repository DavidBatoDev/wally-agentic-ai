# backend/src/dependencies/
"""
Updated dependencies module for the LangGraph-based application.
"""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.langgraph_orchestrator import LangGraphOrchestrator
from src.agent.setup_orchestrator import setup_langgraph_orchestrator
from src.db.db_client import supabase_client
from src.config import get_settings

settings = get_settings()

@lru_cache()
def get_langgraph_orchestrator() -> LangGraphOrchestrator:
    """
    Get or create the LangGraph orchestrator singleton.
    Uses lru_cache to ensure only one instance is created.
    
    Returns:
        The LangGraph orchestrator instance
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.DEFAULT_AGENT_TEMPERATURE,
    )
    
    # Set up the LangGraph orchestrator
    orchestrator = setup_langgraph_orchestrator(
        llm=llm,
        supabase_client=supabase_client
    )
    
    return orchestrator