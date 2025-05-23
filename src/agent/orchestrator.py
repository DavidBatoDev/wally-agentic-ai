# backend/src/agent/orchestrator.py
from typing import Dict, Any, List, Optional, Union
import json
from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor,
)
# from langchain.memory import ConversationBufferMemory # for now we are not using memory
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent.agent_tools import get_tools
from langchain.tools import BaseTool
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.schema import BaseMessage

class AgentOrchestrator:
    """
    Agent orchestrator that processes user messages and manages tools.
    It handles conversation state and produces structured responses using LCEL.
    """
    
    def __init__(self, llm, tools: Optional[List[BaseTool]] = None, db_client=None):
        """
        Initialize the AgentOrchestrator.
        
        Args:
            llm: The LLM instance from Google's Gemini API or similar
            tools: List of LangChain tools the agent can use (if None, tools will be created)
            db_client: Database client for interacting with Supabase
        """
        self.llm = llm
        self.db_client = db_client
        
        # Get tools if not provided
        if tools is None:
            self.tools = get_tools(db_client=db_client)
        else:
            self.tools = tools

        # Create tool names and descriptions for the prompt
        tool_names = ", ".join([tool.name for tool in self.tools])
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # # Use the default structured chat prompt with manual variable substitution
        # base_prompt = StructuredChatAgent.create_prompt(
        #     tools=self.tools,
        #     prefix="Assistant is a large language model trained to be helpful, harmless, and honest.",
        #     suffix="Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation",
        #     human_message_template="{input}\n\n{agent_scratchpad}",
        #     format_instructions=(
        #         "Use a json blob to specify a tool with keys 'action' and 'action_input'.\n\n"
        #         'Valid "action" values: "Final Answer" or {tool_names}\n\n'
        #         "Provide ONLY ONE action per JSON blob."
        #     )
        # )
        
        # The prompt template needs these variables filled manually
        # prompt = base_prompt.partial(
        #     tools=tools_str,
        #     tool_names=tool_names
        # )

        # Create the agent with the custom prompt
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            # prompt=prompt
        )

        # Create the executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,  # Reduced to prevent loops
            max_execution_time=30,  # 30 second timeout
            return_intermediate_steps=True,
        )

    async def _get_conversation_history(self, conversation_id: str) -> List[BaseMessage]:
        """
        Retrieve the conversation history from the database.
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            List of message objects
        """
        # Get recent messages (limit to a reasonable number to avoid context length issues)
        messages = self.db_client.get_conversation_messages(conversation_id, limit=10)

        def _decode_body(body: Union[str, dict]) -> str:
            if isinstance(body, str):
                try:
                    parsed = json.loads(body)
                    return parsed.get("text", body)
                except json.JSONDecodeError:
                    return body
            elif isinstance(body, dict):
                return body.get("text", "")
            else:
                return str(body)
        
        # Convert to LangChain message format
        history = []
        for msg in messages:
            content = _decode_body(msg["body"])
            if msg["sender"] == "user":
                history.append(HumanMessage(content=content))
            elif msg["sender"] == "assistant":
                history.append(AIMessage(content=content))
                    
        return history
    
    async def process_user_message(self, conversation_id: str, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            conversation_id: The ID of the conversation
            user_message: The text message from the user
            context: Additional context like user_id
            
        Returns:
            Response from the assistant
        """
        # Get conversation history
        history = await self._get_conversation_history(conversation_id)
        
        # Run the agent chain
        result = await self.agent_executor.ainvoke({
            "input": user_message,
            "chat_history": history,
        })
        
        final_answer      = result["output"]
        intermediate_steps = result.get("intermediate_steps", [])
        
        # persist the final answer
        assistant_message = self.db_client.create_message(
            conversation_id=str(conversation_id),
            sender="assistant",
            kind="text",
            body=json.dumps({"text": final_answer}),
        )
        
        # If there are intermediate steps, store them for debugging
        if intermediate_steps:
            print("Intermediate steps:", intermediate_steps)
        
        return {
            "message": assistant_message,
            "response": {"kind": "text", "text": final_answer},
        }