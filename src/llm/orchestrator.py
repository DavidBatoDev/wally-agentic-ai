# backend/src/llm/orchestrator.py
"""
LLM orchestrator for managing agent actions and tool calls using LangChain Expression Language.
"""

import json
from typing import Dict, List, Any, Optional, Union, Callable
from uuid import UUID
import traceback
import logging

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool as LangchainTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableConfig

from src.llm.gemini_client import gemini_client
from utils.db_client import supabase_client
from src.services.agent_tools import get_available_tools, ToolRegistry
from src.models.conversation import Message

logger = logging.getLogger(__name__)


class LangChainGeminiModel(BaseChatModel):
    """
    LangChain wrapper for our Gemini client with improved error handling and logging.
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize the LangChain model wrapper."""
        self.client = gemini_client
        self.system_prompt = system_prompt
        super().__init__()
    
    async def _agenerate(
        self, messages: List[Any], stop: Optional[List[str]] = None, 
        run_manager: Optional[BaseCallbackManager] = None, **kwargs
    ) -> Any:
        """Generate a response asynchronously."""
        try:
            # Convert LangChain messages to the format expected by our Gemini client
            prompt = self._convert_messages_to_prompt(messages)
            
            # Generate response
            response = await self.client.generate_text(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            
            return AIMessage(content=response)
            
        except Exception as e:
            logger.error(f"Error in _agenerate: {e}")
            traceback.print_exc()
            # Return a graceful error message
            return AIMessage(content="I encountered an issue processing your request. Please try again.")
    
    def _convert_messages_to_prompt(self, messages: List[Any]) -> str:
        """
        Convert LangChain messages to a prompt string with improved handling of
        different message types including function calls and results.
        """
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                # Handle function calls in AI messages
                if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get('function_call'):
                    func_call = message.additional_kwargs['function_call']
                    prompt_parts.append(f"AI (function call): {func_call['name']}({func_call['arguments']})")
                else:
                    prompt_parts.append(f"AI: {message.content}")
            elif isinstance(message, FunctionMessage):
                # Handle function results
                prompt_parts.append(f"Function {message.name} result: {message.content}")
            else:
                prompt_parts.append(str(message.content))
        
        return "\n".join(prompt_parts)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemini-pro"
    
    @property
    def _identifying_params(self) -> Dict:
        """Return identifying parameters."""
        return {"model_name": "gemini-pro"}


class AgentOrchestrator:
    """
    Orchestrates the agent's actions, memory, and tool use using LangChain Expression Language.
    This improved version focuses on better error handling, memory integration, and state management.
    """
    
    def __init__(self):
        """Initialize the agent orchestrator with improved tool registry handling."""
        self.llm_client = gemini_client
        self.tool_registry = ToolRegistry()
    
    async def _get_conversation_history(self, conversation_id: UUID) -> List[Dict[str, Any]]:
        """
        Retrieve and format conversation history with error handling.
        
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
            
            # Get conversation history
            chat_history = await self._get_conversation_history(conversation_id)
            
            # Get relevant memories if any
            relevant_memories = await self._search_memory(
                conversation_id=conversation_id,
                query=action_message
            )
            
            # Create system prompt with relevant memories
            system_prompt = self._create_base_system_prompt(relevant_memories)
            system_prompt += f"\nThe user has performed action: {action}"
            
            # Get available tools
            tools = self.tool_registry.get_available_tools()
            tool_descriptions = self.tool_registry.get_tool_descriptions()
            
            # Process with LCEL
            response = await self._process_with_lcel(
                conversation_id=conversation_id,
                user_message=action_message,
                system_prompt=system_prompt,
                chat_history=chat_history,
                tools=tools,
                tool_descriptions=tool_descriptions,
                context=context
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in process_action: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your action. Please try again."
            }
    
    async def _store_memory(self, conversation_id: UUID, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Store content in the conversation memory with improved metadata handling.
        
        Args:
            conversation_id: The conversation ID
            content: The text content to store
            meta: Optional metadata
        """
        try:
            # Generate embedding
            embedding = await self.llm_client.get_embeddings(content)
            
            # Store in RAG memory
            supabase_client.store_memory(
                conversation_id=str(conversation_id),
                content=content,
                embedding=embedding,
                meta=meta
            )
            logger.debug(f"Stored memory for conversation {conversation_id}: {content[:50]}...")
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            traceback.print_exc()
    
    async def _search_memory(self, conversation_id: UUID, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation memory with improved error handling.
        
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
    
    def _create_base_system_prompt(self, relevant_memories: List[Dict[str, Any]] = None) -> str:
        """
        Create a base system prompt with optional memory integration.
        
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
        Process a user message using improved LCEL approach.
        
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
            
            # 4. Create chain components using LCEL
            
            # Create system prompt with relevant memories
            system_prompt = self._create_base_system_prompt(relevant_memories)
            
            # Get available tools
            tools = self.tool_registry.get_available_tools()
            tool_descriptions = self.tool_registry.get_tool_descriptions()
            
            # Prepare the agent using new LCEL approach 
            response = await self._process_with_lcel(
                conversation_id=conversation_id,
                user_message=user_message,
                system_prompt=system_prompt,
                chat_history=chat_history,
                tools=tools,
                tool_descriptions=tool_descriptions,
                context=context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_user_message: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your request. Please try again."
            }
    
    async def _process_with_lcel(
        self,
        conversation_id: UUID,
        user_message: str,
        system_prompt: str,
        chat_history: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_descriptions: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a message using LCEL for better composability and transparency.
        
        Args:
            conversation_id: The conversation ID
            user_message: The user's message text
            system_prompt: System prompt with context
            chat_history: Conversation history
            tools: Available tools
            tool_descriptions: Tool descriptions
            context: Additional context
            
        Returns:
            Response dictionary
        """
        try:
            # 1. Generate agent response with tools
            response = await self.llm_client.generate_for_agent(
                prompt=user_message,
                tools=tool_descriptions,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            
            # 2. Handle tool call if present
            if response.get("is_tool_call", False):
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
                supabase_client.create_message(
                    conversation_id=str(conversation_id),
                    sender="assistant",
                    kind="text",
                    body=agent_summary
                )
                
                return {
                    "conversation_id": conversation_id,
                    "response_type": "tool_result",
                    "response": agent_summary,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_result": tool_result
                }
            else:
                # Store assistant response in memory
                await self._store_memory(
                    conversation_id=conversation_id,
                    content=response["text"],
                    meta={"sender": "assistant", "type": "message"}
                )
                
                # Store the assistant's message in the database
                supabase_client.create_message(
                    conversation_id=str(conversation_id),
                    sender="assistant",
                    kind="text",
                    body=response["text"]
                )
                
                return {
                    "conversation_id": conversation_id,
                    "response_type": "text",
                    "response": response["text"]
                }
                
        except Exception as e:
            logger.error(f"Error in _process_with_lcel: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your request. Please try again."
            }
    
    async def create_langchain_agent(
        self, 
        system_prompt: Optional[str] = None,
        conversation_id: Optional[UUID] = None
    ) -> AgentExecutor:
        """
        Create a LangChain agent using our tools and model with LCEL integration.
        
        Args:
            system_prompt: Optional system prompt
            conversation_id: Optional conversation ID for persistent memory
            
        Returns:
            LangChain AgentExecutor
        """
        # Create LangChain-compatible tools
        langchain_tools = []
        
        for tool_name, tool_info in self.tool_registry.get_tools().items():
            # Create a wrapper function that handles async execution
            async def _tool_wrapper(tool_input: Dict[str, Any], tool_name=tool_name, conversation_id=conversation_id):
                try:
                    # Ensure we have conversation_id in the context
                    context = {"conversation_id": str(conversation_id)} if conversation_id else {}
                    
                    # Execute the tool
                    result = await self.tool_registry.execute_tool(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        conversation_id=conversation_id,
                        context=context
                    )
                    
                    # Store tool execution in memory if we have a conversation_id
                    if conversation_id:
                        await self._store_memory(
                            conversation_id=conversation_id,
                            content=f"Tool execution: {tool_name} with result: {json.dumps(result)}",
                            meta={"sender": "system", "type": "tool_execution"}
                        )
                    
                    return json.dumps(result)
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    return json.dumps({"error": str(e)})
            
            # Create a sync wrapper since LangChain expects sync functions
            def make_sync_wrapper(tool_name=tool_name):
                async def async_wrapper(tool_input: str):
                    try:
                        # Parse the input if it's a string
                        if isinstance(tool_input, str):
                            try:
                                parsed_input = json.loads(tool_input)
                            except:
                                parsed_input = {"input": tool_input}
                        else:
                            parsed_input = tool_input
                            
                        # Call the async wrapper
                        return await _tool_wrapper(parsed_input, tool_name=tool_name, conversation_id=conversation_id)
                    except Exception as e:
                        logger.error(f"Error in async wrapper for {tool_name}: {e}")
                        return json.dumps({"error": str(e)})
                
                import asyncio
                def sync_wrapper(tool_input: str):
                    return asyncio.run(async_wrapper(tool_input))
                
                return sync_wrapper
                
            langchain_tool = LangchainTool(
                name=tool_name,
                description=tool_info["description"],
                func=make_sync_wrapper(tool_name),
                args_schema=tool_info["input_schema"],
                return_direct=tool_info.get("return_direct", False)
            )
            langchain_tools.append(langchain_tool)
        
        # Create LangChain-compatible model
        llm = LangChainGeminiModel(system_prompt=system_prompt)
        
        # Create message history from system prompt
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Create the agent
        prompt = ChatPromptTemplate.from_messages(messages + [
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        # Use OpenAI functions agent since it works better with function calling
        agent = create_openai_functions_agent(llm, langchain_tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    async def process_with_langchain(
        self,
        conversation_id: UUID,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using the LangChain LCEL framework.
        
        Args:
            conversation_id: The conversation ID
            user_message: The user's message text
            context: Optional additional context
            
        Returns:
            Response with assistant message and/or next action
        """
        context = context or {}
        
        try:
            # Store user message in memory
            await self._store_memory(
                conversation_id=conversation_id,
                content=user_message,
                meta={"sender": "user", "type": "message"}
            )
            
            # Add conversation_id to context for tools
            context["conversation_id"] = str(conversation_id)
            
            # Get relevant memories
            relevant_memories = await self._search_memory(
                conversation_id=conversation_id,
                query=user_message
            )
            
            # Create system prompt
            system_prompt = self._create_base_system_prompt(relevant_memories)
            
            # Create LangChain agent
            agent_executor = await self.create_langchain_agent(
                system_prompt=system_prompt,
                conversation_id=conversation_id
            )
            
            # Execute agent
            config = RunnableConfig(
                callbacks=None,  # You can add callbacks here if needed
                tags=["conversation", str(conversation_id)],
                metadata={"conversation_id": str(conversation_id)}
            )
            
            result = await agent_executor.ainvoke(
                {
                    "input": user_message,
                    **context
                },
                config=config
            )
            
            # Store assistant response in memory
            await self._store_memory(
                conversation_id=conversation_id,
                content=result["output"],
                meta={"sender": "assistant", "type": "message"}
            )
            
            # Store the assistant's message in the database
            supabase_client.create_message(
                conversation_id=str(conversation_id),
                sender="assistant",
                kind="text",
                body=result["output"]
            )
            
            # Prepare intermediate steps if available
            intermediate_steps = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if hasattr(step, "action") and hasattr(step, "observation"):
                        intermediate_steps.append({
                            "action": str(step.action),
                            "observation": str(step.observation)
                        })
            
            return {
                "conversation_id": conversation_id,
                "response_type": "text",
                "response": result["output"],
                "intermediate_steps": intermediate_steps
            }
        except Exception as e:
            logger.error(f"Error in process_with_langchain: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "response_type": "error",
                "response": "I encountered an error processing your request. Please try again."
            }
    
    async def document_process_chain(
        self,
        conversation_id: UUID,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Specialized chain for document processing workflows using LCEL.
        
        Args:
            conversation_id: The conversation ID
            document_id: The ID of the document to process
            context: Optional additional context
            
        Returns:
            Document processing result
        """
        context = context or {}
        context["document_id"] = document_id
        context["conversation_id"] = str(conversation_id)
        
        try:
            # Create specialized system prompt for document processing
            system_prompt = (
                "You are Wally, an intelligent document processing assistant. "
                "Your task is to analyze the provided document, extract relevant information, "
                "and process it according to the user's needs. Focus on accuracy and completeness "
                "in your extraction and processing."
            )
            
            # Create a specialized agent for document processing
            agent_executor = await self.create_langchain_agent(
                system_prompt=system_prompt,
                conversation_id=conversation_id
            )
            
            # Define document processing steps
            # This uses a sequential chain approach
            document_input = f"Process document with ID: {document_id}"
            
            # Execute agent
            result = await agent_executor.ainvoke(
                {
                    "input": document_input,
                    **context
                }
            )
            
            # Store the processing result in memory
            await self._store_memory(
                conversation_id=conversation_id,
                content=f"Document processing result for document {document_id}: {result['output']}",
                meta={"sender": "system", "type": "document_processing"}
            )
            
            # Store a message about the document processing
            supabase_client.create_message(
                conversation_id=str(conversation_id),
                sender="assistant",
                kind="text",
                body=f"I've processed your document. {result['output']}"
            )
            
            return {
                "conversation_id": conversation_id,
                "document_id": document_id,
                "response_type": "document_processing",
                "response": result["output"]
            }
            
        except Exception as e:
            logger.error(f"Error in document_process_chain: {e}")
            traceback.print_exc()
            return {
                "conversation_id": conversation_id,
                "document_id": document_id,
                "response_type": "error",
                "response": "I encountered an error processing your document. Please try again."
            }


# Singleton instance
agent_orchestrator = AgentOrchestrator()