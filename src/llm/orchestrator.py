# backend/src/llm/orchestrator.py
"""
LLM orchestrator for managing agent actions and tool calls.
"""

import json
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
import traceback
import logging

from src.llm.gemini_client import gemini_client
from src.services.agent_tools import ToolRegistry
from utils.db_client import supabase_client

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates the agent's actions, memory, and tool use.
    This is a simplified version focusing on core functionality.
    """
    
    def __init__(self):
        """Initialize the agent orchestrator."""
        self.llm_client = gemini_client
        self.tool_registry = ToolRegistry()
    
    async def _get_conversation_history(self, conversation_id: UUID) -> List[Dict[str, Any]]:
        """
        Retrieve and format conversation history.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Formatted conversation history
        """
        try:
            messages = supabase_client.get_conversation_messages(str(conversation_id))
            
            history = []
            for msg in messages:
                if msg["kind"] == "text":
                    role = "user" if msg["sender"] == "user" else "assistant"
                    history.append({
                        "role": role,
                        "content": msg["body"]
                    })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            traceback.print_exc()
            return []
        
    async def _store_memory(self, conversation_id: UUID, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Store content in the conversation memory.
        
        Args:
            conversation_id: The conversation ID
            content: The content to store
            meta: Additional metadata
        """
        try:
            # Check if content is too short
            # This is a simple heuristic; you may want to use a more sophisticated check
            if len(content.strip()) < 20:   # tune later
                return
            # Generate embedding
            embedding = await self.llm_client.get_embeddings(content)

            # Store in RAG memory
            supabase_client.store_memory(
                conversation_id=str(conversation_id),
                content=content,
                embedding=embedding,
                meta=meta or {}
            )
            logger.debug(f"Stored memory for conversation {conversation_id}: {content[:50]}...")
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            traceback.print_exc()
    
    async def _search_memory(self, conversation_id: UUID, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation memory.
        
        Args:
            conversation_id: The conversation ID
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of relevant memory entries
        """
        try:
            # Generate embedding for query
            embedding = await self.llm_client.get_embeddings(query)
            
            # Search memory
            results = supabase_client.search_memory(
                conversation_id=str(conversation_id),
                embedding=embedding,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            traceback.print_exc()
            return []
    
    def _create_system_prompt(self, relevant_memories: List[Dict[str, Any]] = None) -> str:
        """
        Create a system prompt with optional memory integration.
        
        Args:
            relevant_memories: List of relevant memory entries
            
        Returns:
            Formatted system prompt
        """
        system_prompt = (
            "You are Wally, an intelligent assistant specialized in document processing, "
            "extraction, and translation. You can help users upload, analyze, extract "
            "information from, and translate documents. Your goal is to be helpful, clear, "
            "and efficient in handling document-related tasks."
        )
        
        if relevant_memories and len(relevant_memories) > 0:
            system_prompt += "\n\nRelevant information from previous conversation:\n"
            for memory in relevant_memories:
                system_prompt += f"- {memory['content']}\n"
                
        return system_prompt
    
    async def process_user_message(
        self,
        conversation_id: UUID,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message.
        
        Args:
            conversation_id: The conversation ID
            user_message: The user's message text
            context: Optional additional context
            
        Returns:
            Response with assistant message and/or next action
        """
        context = context or {}
        
        try:
            # 1. Store user message in memory
            await self._store_memory(
                conversation_id=conversation_id,
                content=user_message,
                meta={"sender": "user", "type": "message"}
            )
            
            # 2. Get conversation history
            chat_history = await self._get_conversation_history(conversation_id)
            
            # 3. Get relevant memories if any
            relevant_memories = await self._search_memory(
                conversation_id=conversation_id,
                query=user_message
            )
            
            # 4. Create system prompt with relevant memories
            system_prompt = self._create_system_prompt(relevant_memories)
            
            # 5. Get tool descriptions
            tool_descriptions = self.tool_registry.get_tool_descriptions()
            
            # 6. Generate agent response with tools
            response = await self.llm_client.generate_for_agent(
                prompt=user_message,
                tools=tool_descriptions,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            
            # 7. Handle tool call if present
            if response.get("is_tool_call", False):
                return await self._handle_tool_call(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    response=response,
                    system_prompt=system_prompt,
                    context=context
                )
            else:
                # 8. If no tool call, just return the text response
                # Store assistant response in memory
                await self._store_memory(
                    conversation_id=conversation_id,
                    content=response["text"],
                    meta={"sender": "assistant", "type": "message"}
                )
                
                # Store the assistant's message in the database
                assistant_message = supabase_client.create_message(
                    conversation_id=str(conversation_id),
                    sender="assistant",
                    kind="text",
                    body=response["text"]
                )
                
                return {
                    "conversation_id": conversation_id,
                    "response_type": "text",
                    "response": response["text"],
                    "message_id": assistant_message.get("id")
                }
                
        except Exception as e:
            logger.error(f"Error in process_user_message: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your request. Please try again."
            }
    
    async def _handle_tool_call(
        self,
        conversation_id: UUID,
        user_message: str,
        response: Dict[str, Any],
        system_prompt: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a tool call from the LLM.
        
        Args:
            conversation_id: The conversation ID
            user_message: The user's message
            response: The LLM response containing the tool call
            system_prompt: The system prompt used
            context: Additional context
            
        Returns:
            Response dictionary
        """
        tool_name = response.get("tool")
        tool_input = response.get("tool_input", {})
        thought = response.get("thought", "")
        
        # Log the tool call
        logger.info(f"Tool call: {tool_name} with input: {json.dumps(tool_input)}")
        
        # Add conversation_id to context if not present
        if "conversation_id" not in context:
            context["conversation_id"] = str(conversation_id)
        
        # Execute the tool
        tool_result = await self.tool_registry.execute_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            conversation_id=conversation_id,
            context=context
        )
        
        # Store tool execution in memory
        await self._store_memory(
            conversation_id=conversation_id,
            content=f"Tool execution: {tool_name} with result: {json.dumps(tool_result)}",
            meta={"sender": "system", "type": "tool_execution"}
        )
        
        # Generate a user-friendly response based on the tool result
        agent_summary = await self.llm_client.generate_text(
            prompt=f"""
            The user said: "{user_message}"
            
            You called the tool "{tool_name}" with the following parameters:
            {json.dumps(tool_input, indent=2)}
            
            The tool returned the following result:
            {json.dumps(tool_result, indent=2)}
            
            Generate a user-friendly response explaining what you found or did.
            """,
            system_prompt=system_prompt
        )
        
        # Store assistant response in memory
        await self._store_memory(
            conversation_id=conversation_id,
            content=agent_summary,
            meta={"sender": "assistant", "type": "message"}
        )
        
        # Store the assistant's message in the database
        assistant_message = supabase_client.create_message(
            conversation_id=str(conversation_id),
            sender="assistant",
            kind="text",
            body=agent_summary
        )
        
        return {
            "conversation_id": conversation_id,
            "response_type": "tool_result",
            "response": agent_summary,
            "message_id": assistant_message.get("id"),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_result": tool_result
        }
    
    async def process_action(
        self,
        conversation_id: UUID,
        action: str,
        values: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an action message (usually from buttons or inputs).
        
        Args:
            conversation_id: The conversation ID
            action: The action identifier
            values: Optional values associated with the action
            context: Optional additional context
            
        Returns:
            Response with assistant message and/or next action
        """
        values = values or {}
        context = context or {}
    
        try:
            # Format action as a structured message
            action_message = f"User action: {action}" + (f" with values {json.dumps(values)}" if values else "")
            
            # Store action in memory
            await self._store_memory(
                conversation_id=conversation_id,
                content=action_message,
                meta={"sender": "user", "type": "action", "action": action, "values": values}
            )
            
            # Process like a regular message
            return await self.process_user_message(
                conversation_id=conversation_id,
                user_message=action_message,
                context={**context, "is_action": True, "action": action, "values": values}
            )
        
        except Exception as e:
            logger.error(f"Error in process_action: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your action. Please try again."
            }


# Singleton instance
agent_orchestrator = AgentOrchestrator()