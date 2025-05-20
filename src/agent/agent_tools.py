# backend/src/agent/agent_tools.py
"""
Tools available to the agent for performing various actions.
"""

from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.agents import Tool
from langchain.tools.base import ToolException

from src.db.db_client import SupabaseClient

def get_tools(db_client: Optional[SupabaseClient] = None) -> List[BaseTool]:
    """
    Get the tools available to the agent.
    
    Args:
        db_client: Database client for tools that need database access
        
    Returns:
        List of LangChain tools
    """
    tools = []
    
    # Basic calculator tool for multiplication
    def multiplication_calculator(a: float, b: float) -> float:
        """
        Multiply two numbers together.
        
        Args:
            a: First number to multiply
            b: Second number to multiply
            
        Returns:
            Product of the two numbers
        """
        return a * b
    
    tools.append(
        Tool(
            name="multiplication_calculator",
            func=multiplication_calculator,
            description="Multiply two numbers together. Input should be two numbers separated by comma.",
            args_schema={
                "a": {"type": "number", "description": "First number to multiply"},
                "b": {"type": "number", "description": "Second number to multiply"}
            }
        )
    )
    
    # Tools that require DB access
    if db_client:
        # Document analysis tool
        def analyze_document(file_id: str, analysis_type: str = "summary") -> Dict[str, Any]:
            """
            Analyze a document to extract insights.
            
            Args:
                file_id: ID of the file to analyze
                analysis_type: Type of analysis to perform (summary, entities, sentiment, etc.)
                
            Returns:
                Analysis results based on the specified type
            """
            try:
                # Get the file object from the database
                file_object = db_client.get_file_object(file_id)
                
                if not file_object:
                    raise ToolException(f"File with ID {file_id} not found")
                
                # Here you would implement the actual document analysis
                # For now, we'll return a placeholder
                return {
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "results": {
                        "summary": "This is a placeholder summary of the document.",
                        "key_points": ["Point 1", "Point 2", "Point 3"],
                        "sentiment": "neutral"
                    }
                }
            except Exception as e:
                raise ToolException(f"Error analyzing document: {str(e)}")
        
        tools.append(
            Tool(
                name="analyze_document",
                func=analyze_document,
                description="Analyze a document to extract insights. Provide the file_id and analysis_type.",
                args_schema={
                    "file_id": {"type": "string", "description": "ID of the file to analyze"},
                    "analysis_type": {"type": "string", "description": "Type of analysis to perform (summary, entities, sentiment, etc.)"}
                }
            )
        )
        
        # Conversation history search tool
        def query_conversation_history(search_term: str, conversation_id: str) -> List[Dict[str, Any]]:
            """
            Query the conversation history to find relevant messages.
            
            Args:
                search_term: Term to search for in the conversation history
                conversation_id: ID of the conversation to search in
                
            Returns:
                List of relevant messages
            """
            try:
                # Get all messages from the conversation
                messages = db_client.get_conversation_messages(conversation_id)
                
                # Filter messages that contain the search term (case-insensitive)
                results = []
                for msg in messages:
                    # Get the message body
                    body = msg.get("body", "")
                    
                    # Check if the search term is in the body
                    if search_term.lower() in body.lower():
                        results.append(msg)
                
                return results
            except Exception as e:
                raise ToolException(f"Error querying conversation history: {str(e)}")
        
        tools.append(
            Tool(
                name="query_conversation_history",
                func=query_conversation_history,
                description="Query the conversation history to find relevant messages. Provide the search_term and conversation_id.",
                args_schema={
                    "search_term": {"type": "string", "description": "Term to search for in the conversation history"},
                    "conversation_id": {"type": "string", "description": "ID of the conversation to search in"}
                }
            )
        )
        
        # Web search tool (mock implementation)
        def web_search(query: str) -> Dict[str, Any]:
            """
            Search the web for information.
            
            Args:
                query: Search query to send to the web search engine
                
            Returns:
                Search results from the web
            """
            # This is a mock implementation
            # In a real implementation, you would connect to a search API
            return {
                "query": query,
                "results": [
                    {
                        "title": "Example Search Result 1",
                        "snippet": "This is an example search result for the query: " + query,
                        "url": "https://example.com/result1"
                    },
                    {
                        "title": "Example Search Result 2",
                        "snippet": "Another example result for: " + query,
                        "url": "https://example.com/result2"
                    }
                ]
            }
        
        tools.append(
            Tool(
                name="web_search",
                func=web_search,
                description="Search the web for information. Provide the search query.",
                args_schema={
                    "query": {"type": "string", "description": "Search query to send to the web search engine"}
                }
            )
        )
    
    return tools