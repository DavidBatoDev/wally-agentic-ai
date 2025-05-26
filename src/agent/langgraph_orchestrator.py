# backend/src/agent/langgraph_orchestrator.py
from typing import Dict, Any, List, Optional
from typing_extensions import Annotated
import json
from dataclasses import dataclass
from enum import Enum

from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
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
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: str
    user_id: str
    current_step: Optional[str]
    workflow_steps: List[WorkflowStep]
    workflow_status: WorkflowStatus
    intermediate_results: Dict[str, Any]
    user_confirmation_pending: bool
    context: Dict[str, Any]
    conversation_history: List[BaseMessage]
    thoughts: Optional[str] = None 


class LangGraphOrchestrator:
    """
    Enhanced orchestrator using LangGraph for multi-step workflow management with memory
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
        """Build the LangGraph workflow (1 LLM call per turn)."""

        workflow = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────
        workflow.add_node("load_memory",          self._load_conversation_memory)
        workflow.add_node("agent",                self._agent_node)
        workflow.add_node("tools",                self._tool_node)
        workflow.add_node("process_tool_results", self._process_tool_results)
        workflow.add_node("finalize",             self._finalize_response)

        # ── Entry ──────────────────────────────────────────────
        workflow.set_entry_point("load_memory")

        # ── Edges ──────────────────────────────────────────────
        workflow.add_edge("load_memory", "agent")                 # memory → agent
        workflow.add_edge("tools", "process_tool_results")        # tools  → results

        # agent decides: call tools or finish
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end":   "finalize",
            },
        )

        # after processing tool output, usually just finish
        workflow.add_conditional_edges(
            "process_tool_results",
            self._after_tool_processing,
            {
                # keep “continue” if you later add looping logic
                "end": "finalize",
            },
        )

        workflow.add_edge("finalize", END)                        # finalize → END

        # ── Compile with in-memory checkpointing ───────────────
        return workflow.compile(checkpointer=self.memory)

    async def _load_conversation_memory(self, state: AgentState) -> AgentState:
        """Load recent conversation history from database"""
        
        if not self.db_client:
            state["conversation_history"] = []
            return state
        
        try:
            # Get last 10 messages from the conversation (excluding the current one)
            recent_messages = self.db_client.get_conversation_messages(
                conversation_id=state["conversation_id"],
                limit=10
            )
            
            # Convert to LangChain message format
            history_messages = []
            for msg in recent_messages:
                try:
                    body_data = json.loads(msg["body"])
                    content = body_data.get("text", str(body_data))
                    
                    if msg["sender"] == "user":
                        history_messages.append(HumanMessage(content=content))
                    elif msg["sender"] == "assistant":
                        history_messages.append(AIMessage(content=content))
                        
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed messages
                    continue
            
            state["conversation_history"] = history_messages
            print(f"Loaded {len(history_messages)} messages from conversation history")
            
        except Exception as e:
            print(f"Error loading conversation memory: {e}")
            state["conversation_history"] = []
        
        return state
    
    def _prepare_messages_for_llm(self, state: AgentState) -> List[BaseMessage]:
        """Prepare messages for LLM including conversation history"""
        
        # Start with conversation history
        all_messages = state["conversation_history"].copy()

        # Add current conversation messages (excluding tool messages)
        current_messages = []
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                current_messages.append(msg)
        
        # Combine history and current messages
        all_messages.extend(current_messages)
        
        # Limit total context to reasonable size (last 20 messages)
        if len(all_messages) > 20:
            all_messages = all_messages[-20:]
        
        return all_messages
    
    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent processing node with proper message handling"""
        
        try:
            # Prepare messages for LLM (includes history + current conversation)
            # llm_messages = self._prepare_messages_for_llm(state)
            llm_messages = [
                SystemMessage(
                    content=(
                        "You are an assistant with access to tools. "
                        "First think step-by-step **internally**. "
                        "If a tool call is needed, respond with the JSON tool call only. "
                        "Otherwise answer the user. Never expose your private thoughts."
                   )
               ),
               *self._prepare_messages_for_llm(state),
            ]

            # DEBUG: Print messages being sent to LLM
            print("🧠  _agent_node – sending to LLM")
            for m in llm_messages:
                print("   ", type(m).__name__, "→", m.content[:120])

            if not llm_messages:
                # Create a default response if no messages
                default_response = AIMessage(content="I'm ready to help you with your request.")
                state["messages"].append(default_response)
                state["workflow_status"] = WorkflowStatus.COMPLETED
                return state
            
            # Call the LLM with tools
            response = await self.llm_with_tools.ainvoke(llm_messages)

            # DEBUG: Print the last message before invoking LLM
            print("🧠  _agent_node – LLM replied:", response.content[:120] if hasattr(response, "content") else response)

            # DEBUG: Print the response from LLM
            if getattr(response, "tool_calls", None):
                for call in response.tool_calls:
                    print("   ↳ tool requested:", call["name"], call["args"])
            
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
    
    async def _tool_node(self, state: AgentState) -> AgentState:
        """Execute tools and handle their responses"""
        
        try:
            # Get the last message which should have tool calls
            messages = state["messages"]
            if not messages:
                return state
                
            last_message = messages[-1]

            # DEBUG: Print the last message before executing tools
            for call in last_message.tool_calls:
                print(f"🔧  calling {call['name']} with", call['args'])

            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            print(f"Executing {len(last_message.tool_calls)} tool calls")
            
            # Execute tools using LangGraph's ToolNode
            tool_response = await self.tool_node.ainvoke(state)
            print("🔧  tool(s) finished")
            
            # The tool node updates the state with tool messages
            return tool_response
            
        except Exception as e:
            print(f"Error in tool execution: {e}")
            # Add error message
            error_tool_message = ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id="error" if not last_message.tool_calls else last_message.tool_calls[0].get("id", "error")
            )
            state["messages"].append(error_tool_message)
            return state
    
    async def _process_tool_results(self, state: AgentState) -> AgentState:
        """Process tool results and prepare final agent response"""
        
        messages = state["messages"]
        if not messages:
            return state
        
        # Check if any UI tools were used
        ui_tool_used = False
        tool_results = []
        
        # Look for recent tool messages
        for message in reversed(messages[-5:]):
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card"]:
                        ui_tool_used = True
                        print(f"UI tool used: {content.get('kind')}")
                        break
                except (json.JSONDecodeError, AttributeError):
                    # Regular tool result
                    tool_results.append(message.content)
        
        # Store whether UI tool was used for decision making
        state["context"]["ui_tool_used"] = ui_tool_used
        
        # If no UI tool was used and we have tool results, prepare a response
        if not ui_tool_used and tool_results:
            # Create a summarizing message for the agent to respond with
            tool_summary = ". ".join(tool_results)
            summary_message = AIMessage(
                content=f"Based on the calculation: {tool_summary}"
            )
            state["messages"].append(summary_message)
            print(f"Added summary message: {summary_message.content}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to tools or end"""
        
        messages = state["messages"]
        if not messages:
            return "end"
            
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def _after_tool_processing(self, state: AgentState) -> str:
        """Decide what to do after tool processing"""
        
        # If a UI tool was used, end the conversation
        if state["context"].get("ui_tool_used", False):
            return "end"
        
        # For regular tools (like math), always end after processing
        # The summary response has been added, no need to continue
        return "end"
    
    async def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response"""
        
        messages = state["messages"]
        
        # Look for UI tool responses in the message history
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "notification"]:
                        # Use the UI tool response as the final response
                        state["context"]["final_response"] = content
                        state["workflow_status"] = WorkflowStatus.COMPLETED
                        return state
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # If no UI tool response found, use the last AI message
        if messages:
            # Look for the last AI message
            for message in reversed(messages):
                if isinstance(message, AIMessage) and hasattr(message, 'content') and message.content:
                    final_response = {
                        "kind": "text",
                        "text": message.content
                    }
                    break
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
            context=context,
            conversation_history=[],
        )
        
        # Create thread config for state persistence
        config = RunnableConfig(
            configurable={"thread_id": conversation_id}
        )
        
        # Execute the workflow
        try:
            print(f"Starting workflow for conversation: {conversation_id}")
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Get the final response
            final_response = final_state["context"].get("final_response")
            
            # If no final response was set, extract from the last message
            if not final_response:
                messages = final_state["messages"]
                if messages:
                    # Find the last AI message
                    for message in reversed(messages):
                        if isinstance(message, AIMessage) and hasattr(message, 'content') and message.content:
                            final_response = {
                                "kind": "text",
                                "text": message.content
                            }
                            break
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
            
            print(f"Final response: {final_response}")

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
                "steps_completed": len([msg for msg in final_state["messages"] if isinstance(msg, AIMessage)])
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