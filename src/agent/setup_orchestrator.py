# backend/src/agent/setup_orchestrator.py
"""
Setup module for the LangGraph orchestrator with all necessary configurations.
"""

from typing import Dict, Any, List, Optional
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

from src.agent.langgraph_orchestrator import LangGraphOrchestrator
from src.agent.agent_tools import get_tools
from src.db.db_client import SupabaseClient
from src.config import get_settings

def setup_langgraph_orchestrator(
    llm: BaseLanguageModel,
    supabase_client: SupabaseClient,
    config: Optional[Dict[str, Any]] = None
) -> LangGraphOrchestrator:
    """
    Set up the LangGraph orchestrator with all necessary tools and configurations.
    
    Args:
        llm: Language model to use for the orchestrator
        supabase_client: Supabase client for database access
        config: Optional configuration options
        
    Returns:
        Configured LangGraphOrchestrator instance
    """
    # Initialize default config if not provided
    if config is None:
        config = {}
    
    # Get tools from agent_tools module
    tools = get_tools(db_client=supabase_client)
    
    # Create the LangGraph orchestrator
    orchestrator = LangGraphOrchestrator(
        llm=llm,
        tools=tools,
        db_client=supabase_client
    )
    
    return orchestrator