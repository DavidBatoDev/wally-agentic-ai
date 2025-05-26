# backend/src/agent/langgraph_orchestrator.py
from typing import Dict, Any, List, Optional, Union, Literal
import json
from dataclasses import dataclass
from enum import Enum

from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.agent.agent_tools import get_tools
from src.db.db_client import SupabaseClient


class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    id: str
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    requires_confirmation: bool = False


class AgentState(Dict):
    """State for the LangGraph workflow"""
    messages: List[BaseMessage]
    conversation_id: str
    user_id: str
    current_step: Optional[str]
    workflow_steps: List[WorkflowStep]
    workflow_status: WorkflowStatus
    intermediate_results: Dict[str, Any]
    user_confirmation_pending: bool
    context: Dict[str, Any]


class LangGraphOrchestrator:
    """
    Enhanced orchestrator using LangGraph for multi-step workflow management
    """
    
    def __init__(self, llm, tools: Optional[List[BaseTool]] = None, db_client: Optional[SupabaseClient] = None):
        self.llm = llm
        self.db_client = db_client
        
        # Get tools if not provided
        if tools is None:
            self.tools = get_tools(db_client=db_client)
        else:
            self.tools = tools
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create tool node for LangGraph
        self.tool_node = ToolNode(self.tools)
        
        # Initialize memory for state persistence
        self.memory = MemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the simplified LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("finalize", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "finalize"
            }
        )
        
        # After tools, go back to agent for next iteration
        workflow.add_edge("tools", "agent")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _filter_messages_for_llm(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Filter and prepare messages for LLM consumption.
        Handle ToolMessages that should end the conversation.
        """
        filtered_messages = []
        
        for message in messages:
            if isinstance(message, ToolMessage):
                # Check if this is a UI tool that should end the conversation
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "progress", "notification"]:
                        # This is a UI response that should end the conversation
                        # Don't include it in LLM messages, but store it for final response
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
                
                # For other tool messages, convert to a regular message
                filtered_messages.append(AIMessage(
                    content=f"Tool result: {message.content}"
                ))
            elif isinstance(message, (HumanMessage, AIMessage)):
                # Only include messages with actual content
                if hasattr(message, 'content') and message.content and message.content.strip():
                    filtered_messages.append(message)
        
        return filtered_messages
    
    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent processing node with proper message handling"""
        
        try:
            # Get the conversation history
            messages = state["messages"]
            
            # Check if the last message is a UI tool response
            if messages and isinstance(messages[-1], ToolMessage):
                try:
                    content = json.loads(messages[-1].content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "progress", "notification"]:
                        # This is a UI response - conversation should end here
                        state["workflow_status"] = WorkflowStatus.COMPLETED
                        return state
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Filter messages for LLM consumption
            llm_messages = self._filter_messages_for_llm(messages)
            
            # If no valid messages for LLM, create a default response
            if not llm_messages:
                # This shouldn't happen in normal flow, but handle it gracefully
                default_response = AIMessage(content="I'm ready to help you with your request.")
                state["messages"].append(default_response)
                state["workflow_status"] = WorkflowStatus.COMPLETED
                return state
            
            print(f"Sending to LLM: {[msg.content for msg in llm_messages]}")
            
            # Call the LLM with tools
            response = await self.llm_with_tools.ainvoke(llm_messages)
            
            # Add the response to messages
            state["messages"].append(response)
            
            # Update status
            state["workflow_status"] = WorkflowStatus.IN_PROGRESS
            
        except Exception as e:
            print(f"Error in agent node: {e}")
            # Create error response
            error_message = AIMessage(content=f"I encountered an error: {str(e)}")
            state["messages"].append(error_message)
            state["workflow_status"] = WorkflowStatus.FAILED
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to tools or end"""
        
        messages = state["messages"]
        if not messages:
            return "end"
            
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    async def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response"""
        
        messages = state["messages"]
        
        # Look for UI tool responses in the message history
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "progress", "notification"]:
                        # Use the UI tool response as the final response
                        state["context"]["final_response"] = content
                        state["workflow_status"] = WorkflowStatus.COMPLETED
                        return state
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # If no UI tool response found, use the last AI message
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                final_response = {
                    "kind": "text",
                    "text": last_message.content
                }
            else:
                final_response = {
                    "kind": "text",
                    "text": "Task completed."
                }
        else:
            final_response = {
                "kind": "text",
                "text": "I'm ready to help you."
            }
        
        state["context"]["final_response"] = final_response
        state["workflow_status"] = WorkflowStatus.COMPLETED
        
        return state
    
    async def process_user_message(self, conversation_id: str, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message using the LangGraph workflow
        """
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_message)],
            conversation_id=conversation_id,
            user_id=context.get("user_id", ""),
            current_step=None,
            workflow_steps=[],
            workflow_status=WorkflowStatus.PENDING,
            intermediate_results={},
            user_confirmation_pending=False,
            context=context
        )
        
        # Create thread config for state persistence
        config = RunnableConfig(
            configurable={"thread_id": conversation_id}
        )
        
        # Execute the workflow
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Get the final response
            final_response = final_state["context"].get("final_response")
            
            # If no final response was set, extract from the last message
            if not final_response:
                messages = final_state["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        final_response = {
                            "kind": "text",
                            "text": last_message.content
                        }
                    else:
                        final_response = {
                            "kind": "text",
                            "text": "I processed your request."
                        }
                else:
                    final_response = {
                        "kind": "text",
                        "text": "I apologize, but I couldn't process your request."
                    }
            
            # Store the assistant message
            assistant_message = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind=final_response["kind"],
                body=json.dumps(final_response),
            )
            
            return {
                "message": assistant_message,
                "response": final_response,
                "workflow_status": final_state["workflow_status"].value,
                "steps_completed": 1
            }
            
        except Exception as e:
            print(f"Error in workflow execution: {e}")
            # Handle workflow execution errors
            error_response = {
                "kind": "text",
                "text": f"I encountered an error while processing your request: {str(e)}"
            }
            
            assistant_message = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps(error_response),
            )
            
            return {
                "message": assistant_message,
                "response": error_response,
                "workflow_status": WorkflowStatus.FAILED.value,
                "steps_completed": 0
            }
    
    async def handle_user_action(self, conversation_id: str, action: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user actions (like button clicks) in ongoing workflows
        """
        
        # For now, return a simple response
        # In the future, you can implement more complex action handling
        response = {
            "kind": "text",
            "text": f"You selected: {action}. Action processed successfully."
        }
        
        assistant_message = self.db_client.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps(response),
        )
        
        return {
            "message": assistant_message,
            "response": response,
            "workflow_status": WorkflowStatus.COMPLETED.value
        }