# backend/src/agent/setup_orchestrator.py
from typing import Dict, Any, List, Union, Optional
from langchain.memory import ConversationBufferMemory
from langchain.schema.language_model import BaseLanguageModel
from pathlib import Path
import os
import json

from src.agent.orchestrator import AgentOrchestrator
from src.agent.agent_tools import get_tools
from src.db.db_client import SupabaseClient
from src.config import get_settings

def setup_agent_orchestrator(
    llm: BaseLanguageModel, 
    supabase_client: SupabaseClient,
    config: Dict[str, Any] = None
) -> AgentOrchestrator:
    """
    Set up the agent orchestrator with all necessary tools.
    
    Args:
        llm: Language model to use for the agent
        supabase_client: Supabase client for database access
        config: Configuration options
        
    Returns:
        Configured AgentOrchestrator instance
    """
    # Initialize default config if not provided
    if config is None:
        config = {}
    
    # Get tools from agent_tools module    
    tools = get_tools(db_client=supabase_client)    
    # Create the agent orchestrator with specified agent type
    orchestrator = AgentOrchestrator(
        llm=llm, 
        tools=tools, 
        db_client=supabase_client
    )
    
    return orchestrator

def load_system_prompt(prompt_name: str) -> str:
    """
    Load a system prompt template from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without extension)
        
    Returns:
        The prompt template as a string
    """
    # Get the base directory of the application
    base_dir = Path(__file__).parent.parent.parent
    
    # Path to the prompts directory
    prompts_dir = base_dir / "prompts"
    
    # Path to the specific prompt file
    prompt_path = prompts_dir / f"{prompt_name}.txt"
    
    # Check if the file exists
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    # Load the prompt template
    with open(prompt_path, "r") as file:
        prompt_template = file.read()
    
    return prompt_template

def create_tool_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Create descriptions for all tools that can be used in the system.
    This is useful for dynamic tool selection.
    
    Returns:
        Dictionary of tool descriptions
    """
    return {
        "multiplication_calculator": {
            "name": "multiplication_calculator",
            "description": "Multiply two numbers together",
            "parameters": {
                "a": "First number to multiply",
                "b": "Second number to multiply"
            },
            "output": "Product of the two numbers"
        },
        "analyze_document": {
            "name": "analyze_document",
            "description": "Analyze a document to extract insights",
            "parameters": {
                "file_id": "ID of the file to analyze",
                "analysis_type": "Type of analysis to perform (summary, entities, sentiment, etc.)"
            },
            "output": "Analysis results based on the specified type"
        },
        "query_conversation_history": {
            "name": "query_conversation_history",
            "description": "Query the conversation history to find relevant messages",
            "parameters": {
                "search_term": "Term to search for in the conversation history",
                "conversation_id": "ID of the conversation to search in"
            },
            "output": "Relevant messages from the conversation history"
        },
        "web_search": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": "Search query to send to the web search engine"
            },
            "output": "Search results from the web"
        }
    }

# Sample implementation of the main application setup
# def setup_application(config_path: Optional[str] = None) -> Dict[str, Any]:
#     """
#     Set up the main application components.
    
#     Args:
#         config_path: Path to the configuration file
        
#     Returns:
#         Dictionary containing the configured components
#     """
#     # Get settings from config module
#     settings = get_settings()
    
#     # Load additional configuration if provided
#     config = {}
#     if config_path and os.path.exists(config_path):
#         with open(config_path, "r") as file:
#             config = json.load(file)
    
#     # Initialize Supabase client
#     supabase_url = config.get("supabase_url", settings.SUPABASE_URL)
#     supabase_key = config.get("supabase_key", settings.SUPABASE_KEY)
#     supabase_client = SupabaseClient(url=supabase_url, key=supabase_key)
    
#     # Initialize the LLM with appropriate temperature for agentic tasks
#     from langchain_google_genai import ChatGoogleGenerativeAI
    
#     google_api_key = config.get("google_api_key", settings.GEMINI_API_KEY)
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         google_api_key=google_api_key,
#         temperature=0.2,  # Lower temperature for more deterministic tool use
#         streaming=True
#     )
    
#     # Set up the agent orchestrator with STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
#     agent_orchestrator = setup_agent_orchestrator(
#         llm=llm, 
#         supabase_client=supabase_client,
#         config=config.get("agent_config", {})
#     )
    
#     return {
#         "supabase_client": supabase_client,
#         "llm": llm,
#         "agent_orchestrator": agent_orchestrator,
#         "config": config
#     }