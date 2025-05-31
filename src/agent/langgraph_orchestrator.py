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
from src.db.checkpointer import supabase_checkpointer


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
    conversation_context: Optional[str] = None


class LangGraphOrchestrator:
    """
    Enhanced orchestrator using LangGraph for multi-step workflow management with Supabase buffer memory
    """
    
    def __init__(self, llm, tools: Optional[List[BaseTool]] = None, db_client: Optional[SupabaseClient] = None):
        self.llm = llm
        self.db_client = db_client
        self.checkpointer = supabase_checkpointer
        
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
        workflow.add_node("load_buffer",          self._load_conversation_buffer)
        workflow.add_node("agent",                self._agent_node)
        workflow.add_node("tools",                self._tool_node)
        workflow.add_node("process_tool_results", self._process_tool_results)
        workflow.add_node("finalize",             self._finalize_response)
        workflow.add_node("save_buffer",          self._save_conversation_buffer)

        # ── Entry ──────────────────────────────────────────────
        workflow.set_entry_point("load_buffer")

        # ── Edges ──────────────────────────────────────────────
        workflow.add_edge("load_buffer", "agent")                 # buffer → agent
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
                "end": "finalize",
            },
        )

        workflow.add_edge("finalize", "save_buffer")              # finalize → save
        workflow.add_edge("save_buffer", END)                     # save → END

        # ── Compile with in-memory checkpointing ───────────────
        return workflow.compile(checkpointer=self.memory)

    async def _load_conversation_buffer(self, state: AgentState) -> AgentState:
        """Load conversation buffer from Supabase using the enhanced checkpointer"""

        try:
            conversation_id = state["conversation_id"]
            
            # Load buffer from Supabase
            buffer_messages = self.checkpointer.load_buffer(conversation_id)
            
            print(f"🔍 _load_conversation_buffer → loaded {len(buffer_messages)} messages from buffer")
            
            # Set the conversation history from buffer
            state["conversation_history"] = buffer_messages
            
            # Get enhanced buffer info with tool usage analytics
            buffer_info = self.checkpointer.get_buffer_info(conversation_id)
            if buffer_info:
                analytics = buffer_info.get('buffer_analytics', {})
                print(f"Buffer info - Size: {buffer_info['buffer_size']}, Last updated: {buffer_info['updated_at']}")
                
                # Log tool usage summary
                tool_usage = analytics.get('tool_usage', {})
                if tool_usage:
                    print(f"Tool usage - Messages with tools: {tool_usage.get('messages_with_tools', 0)}, "
                            f"Total tool calls: {tool_usage.get('tool_calls_count', 0)}")
                    if tool_usage.get('tools_used'):
                        print(f"Tools used in conversation: {list(tool_usage['tools_used'].keys())}")
                
                # Store analytics in state for reference
                state["context"]["buffer_analytics"] = analytics
            
            print(f"Loaded {len(buffer_messages)} messages from conversation buffer")
            
            # Clear any conversation context for fresh processing
            state["conversation_context"] = None
            
        except Exception as e:
            print(f"Error loading conversation buffer: {e}")
            state["conversation_history"] = []

        return state

    def _prepare_messages_for_llm(self, state: AgentState) -> List[BaseMessage]:
        """Prepare messages for LLM including conversation history from buffer"""
        
        # Start with conversation history from buffer
        all_messages = state["conversation_history"].copy()

        # Add current conversation messages (excluding tool messages and avoiding duplicates)
        current_messages = []
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                # Avoid adding duplicate messages that are already in buffer
                msg_content = getattr(msg, "content", "").strip()
                if msg_content and not any(
                    getattr(hist_msg, "content", "").strip() == msg_content 
                    for hist_msg in all_messages
                ):
                    current_messages.append(msg)
        
        # Combine buffer history and current messages
        all_messages.extend(current_messages)
        
        # Limit total context to reasonable size (last 20 messages for better context)
        if len(all_messages) > 20:
            all_messages = all_messages[-20:]

        # Filter out empty messages
        filtered = [m for m in all_messages if getattr(m, "content", "").strip()]
        
        print(f"🔍 _prepare_messages_for_llm → sending {len(filtered)} messages to LLM")
        for i, m in enumerate(filtered):
            print(f"   [{i:2d}] {type(m).__name__}: {repr(m.content[:60])}...")
        
        return filtered
    
    def _create_system_message(self, state: AgentState) -> str:
        """Create a context-aware system message with tool usage history."""
        
        base_prompt = (
            "You are a helpful assistant specialized in editing, formatting, and translating documents. "
            "Always use tools to assist with tasks like buttons for asking for options for follow up questions, upload buttons for file uploads, and calculation for solving maths. "
            "You have access to tools to support these tasks effectively. "
            "Always reason step-by-step **internally** before responding. "
            "If a tool call is needed (e.g., for translation, formatting, or text extraction), respond with the JSON tool call only. "
            "Otherwise, reply directly to the user. "
            "Never reveal your internal thoughts or reasoning process. "
            "Focus on clarity, precision, and professionalism in all responses."
        )
        
        # Enhance with conversation context if available
        conversation_id = state["conversation_id"]
        context_summary = self.checkpointer.get_buffer_context_summary(conversation_id)
        
        if context_summary:
            tool_usage = context_summary.get("tool_usage_summary", {})
            recent_context = context_summary.get("recent_context", {})
            
            # Add context about recent tool usage
            if tool_usage.get("tools_used"):
                tools_used = ", ".join(tool_usage["tools_used"].keys())
                base_prompt += f"\n\nContext: In this conversation, you have used these tools: {tools_used}. "
                
                # Add specific context based on tools used
                if "show_buttons" in tool_usage["tools_used"]:
                    base_prompt += "You've presented options to the user before. "
                if "calculate" in tool_usage["tools_used"] or "multiply_numbers" in tool_usage["tools_used"]:
                    base_prompt += "You've performed calculations in this conversation. "
                if "show_upload_button" in tool_usage["tools_used"]:
                    base_prompt += "You've provided file upload functionality. "
            
            # Add recent conversation flow context
            if recent_context.get("last_tool_used"):
                last_tool = recent_context["last_tool_used"]
                base_prompt += f"The most recent tool used was '{last_tool['name']}' with kind '{last_tool['kind']}'. "
                
                # Avoid repeating the same tool unnecessarily
                if last_tool["name"] in ["show_buttons", "show_upload_button"]:
                    base_prompt += "Avoid immediately using the same UI tool unless specifically requested. "
            
            # Add quality assessment
            quality = context_summary.get("context_quality", {})
            if quality.get("recommendation"):
                base_prompt += f"Conversation quality: {quality['recommendation']}. "
        
        return base_prompt
    
    def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get insights about the conversation including tool usage patterns.
        """
        try:
            context_summary = self.checkpointer.get_buffer_context_summary(conversation_id)
            if not context_summary:
                return {"error": "No conversation data found"}
            
            tool_usage = context_summary.get("tool_usage_summary", {})
            recent_context = context_summary.get("recent_context", {})
            quality = context_summary.get("context_quality", {})
            
            insights = {
                "conversation_health": {
                    "quality_score": quality.get("score", 0),
                    "recommendation": quality.get("recommendation", "Unknown"),
                    "factors": quality.get("factors", [])
                },
                "tool_usage_patterns": {
                    "total_tool_calls": tool_usage.get("tool_calls_count", 0),
                    "messages_with_tools": tool_usage.get("messages_with_tools", 0),
                    "tools_breakdown": tool_usage.get("tools_used", {}),
                    "ui_tools_used": tool_usage.get("ui_tools_used", 0),
                    "calculation_tools_used": tool_usage.get("calculation_tools_used", 0)
                },
                "conversation_flow": {
                    "last_user_input": recent_context.get("last_user_message", "No recent input"),
                    "last_assistant_response": recent_context.get("last_assistant_message", "No recent response"),
                    "last_tool_interaction": recent_context.get("last_tool_used", "No recent tool use")
                },
                "buffer_info": {
                    "total_messages": context_summary.get("buffer_size", 0),
                    "message_distribution": context_summary.get("message_distribution", {}),
                    "last_updated": context_summary.get("updated_at", "Unknown")
                }
            }
            
            return insights
            
        except Exception as e:
            return {"error": f"Failed to get conversation insights: {str(e)}"}

    
    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent processing node with proper message handling"""
        
        try:
            # Prepare messages for LLM (includes buffer history + current conversation)
            llm_messages = [
                SystemMessage(content=self._create_system_message(state)),
                *self._prepare_messages_for_llm(state),
            ]

            if not llm_messages:
                # Create a default response if no messages
                default_response = AIMessage(content="I'm ready to help you with your request.")
                state["messages"].append(default_response)
                state["workflow_status"] = WorkflowStatus.COMPLETED
                return state
            
            print("➤ Payload being sent to Gemini:")
            for idx, m in enumerate(llm_messages):
                content = getattr(m, "content", None)
                print(f"  [{idx:2d}] ({type(m).__name__}): {repr(content)}  length={len(content or '')}")
            
            # Call the LLM with tools
            response = await self.llm_with_tools.ainvoke(llm_messages)

            print("🧠  _agent_node – LLM replied:", response.content[:120] if hasattr(response, "content") else response)

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
        
        # Check if any UI tools were used in the CURRENT interaction
        ui_tool_used = False
        tool_results = []
        
        # Only look at the most recent tool messages (last 3 messages)
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        for message in recent_messages:
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "upload_button"]:
                        ui_tool_used = True
                        print(f"UI tool used in current cycle: {content.get('kind')}")
                        break
                except (json.JSONDecodeError, AttributeError):
                    # Regular tool result
                    if hasattr(message, 'content'):
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
        
        # Check if a UI tool was used in the current cycle
        if state["context"].get("ui_tool_used", False):
            print("Ending after UI tool usage")
            return "end"
        
        # For regular tools (like math), always end after processing
        print("Ending after regular tool processing")
        return "end"
    
    async def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response - only use UI tools from current workflow cycle"""
        
        messages = state["messages"]
        
        # Only look for UI tool responses from the CURRENT workflow cycle
        current_cycle_ui_response = None
        
        # Look for UI tool responses, but only from recent messages (last 3-5 messages)
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        for message in reversed(recent_messages):
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "upload_button"]:
                        current_cycle_ui_response = content
                        print(f"Found recent UI tool response: {content.get('kind')}")
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # Only use UI response if it's from the current cycle AND we actually called a tool
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break
        
        use_ui_response = (
            current_cycle_ui_response and 
            last_ai_message and 
            hasattr(last_ai_message, 'tool_calls') and 
            last_ai_message.tool_calls and
            not state["context"].get("force_text_response", False)
        )
        
        if use_ui_response:
            state["context"]["final_response"] = current_cycle_ui_response
            state["workflow_status"] = WorkflowStatus.COMPLETED
            print(f"Using UI response: {current_cycle_ui_response.get('kind')}")
            return state
        
        # Otherwise, use the last AI message as text response
        if messages:
            # Look for the last AI message with actual content
            for message in reversed(messages):
                if isinstance(message, AIMessage) and hasattr(message, 'content') and message.content.strip():
                    final_response = {
                        "kind": "text",
                        "text": message.content
                    }
                    print(f"Using text response: {message.content[:50]}...")
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
    
    async def _save_conversation_buffer(self, state: AgentState) -> AgentState:
        """Save the conversation messages to buffer using the enhanced checkpointer"""
        
        try:
            conversation_id = state["conversation_id"]
            
            # Collect all meaningful messages from the conversation
            buffer_messages = []
            
            # Add messages from history (already in buffer format)
            buffer_messages.extend(state["conversation_history"])
            
            # Add new messages from current conversation, including tool messages
            for msg in state["messages"]:
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    # Avoid duplicates
                    msg_content = getattr(msg, "content", "").strip()
                    
                    # For tool messages, always include them as they provide important context
                    if isinstance(msg, ToolMessage):
                        buffer_messages.append(msg)
                    elif msg_content and not any(
                        getattr(existing_msg, "content", "").strip() == msg_content 
                        for existing_msg in buffer_messages
                        if not isinstance(existing_msg, ToolMessage)
                    ):
                        buffer_messages.append(msg)
            
            # Trim buffer to reasonable size (keep last 50 messages)
            if len(buffer_messages) > 50:
                buffer_messages = buffer_messages[-50:]
            
            # Save to buffer with enhanced analytics
            success = self.checkpointer.save_buffer(conversation_id, buffer_messages)
            
            if success:
                print(f"✅ Successfully saved {len(buffer_messages)} messages to buffer")
                
                # Log buffer analytics for debugging
                context_summary = self.checkpointer.get_buffer_context_summary(conversation_id)
                if context_summary:
                    quality = context_summary.get("context_quality", {})
                    print(f"Buffer quality score: {quality.get('score', 0)}/100 - {quality.get('recommendation', 'N/A')}")
            else:
                print("❌ Failed to save messages to buffer")
            
            # Update state with final buffer
            state["conversation_history"] = buffer_messages
            
        except Exception as e:
            print(f"Error saving conversation buffer: {e}")
        
        return state


    async def process_user_message(self, conversation_id: str, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message using the LangGraph workflow with buffer management
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

            # Store the assistant message in database
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
        Handle user actions (like button clicks) in ongoing workflows.
        Enhanced to preserve context and prevent repetitive questioning.
        """
        
        try:
            print(f"Handling user action: {action} with values: {values}")
            
            if action.startswith("button_") or action in ["yes", "no", "confirm", "cancel"]:
                # Extract the button information
                button_text = None
                
                # Try to get the button text from values
                if "label" in values:
                    button_text = values["label"]
                elif "value" in values:
                    button_text = values["value"]
                elif "text" in values:
                    button_text = values["text"]
                else:
                    # Fallback to the action name, cleaned up
                    button_text = action.replace("button_", "").replace("_", " ").title()
                
                # Create a natural response based on the button selection
                contextual_message = f"My answer is: {button_text}"
                
                # Enhanced context with button selection info
                context = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "button_context": {
                        "selected_text": button_text,
                        "selected_action": action,
                        "is_answer": True  # Mark this as an answer to prevent re-asking
                    }
                }
                
                workflow_response = await self.process_user_message(
                    conversation_id=conversation_id,
                    user_message=contextual_message,
                    context=context
                )
                
                return workflow_response
            
            # Handle file upload actions
            elif action == "file_uploaded":
                # Extract file information
                file_info = values.get("file_info", {})
                file_name = file_info.get("name", "Unknown file")
                file_size = file_info.get("size", "Unknown size")
                file_type = file_info.get("type", "Unknown type")
                
                # Create a descriptive message about the file upload
                upload_message = f"I uploaded a file: {file_name} ({file_size}, {file_type})"
                
                print(f"Processing file upload as text message: '{upload_message}'")
                
                context = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "file_info": file_info
                }
                
                workflow_response = await self.process_user_message(
                    conversation_id=conversation_id,
                    user_message=upload_message,
                    context=context
                )
                
                return workflow_response
            
            # Handle form submissions from show_inputs tool
            elif action == "form_submitted":
                # Convert form data to a readable message
                form_data = values.get("form_data", {})
                
                # Create a message describing the form submission
                form_message_parts = []
                for field_name, field_value in form_data.items():
                    form_message_parts.append(f"{field_name}: {field_value}")
                
                form_message = "I submitted the form with: " + ", ".join(form_message_parts)
                
                print(f"Processing form submission as text message: '{form_message}'")
                
                context = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "form_data": form_data
                }
                
                workflow_response = await self.process_user_message(
                    conversation_id=conversation_id,
                    user_message=form_message,
                    context=context
                )
                
                return workflow_response
            
            # Handle unknown actions with a generic approach
            else:
                print(f"Unknown action type: {action}, treating as generic text")
                
                # Try to create a meaningful message from the action
                if values.get("message"):
                    action_message = values["message"]
                else:
                    action_message = f"I selected: {action}"
                
                context = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "action_values": values
                }
                
                workflow_response = await self.process_user_message(
                    conversation_id=conversation_id,
                    user_message=action_message,
                    context=context
                )
                
                return workflow_response
            
        except Exception as e:
            print(f"Error handling user action: {e}")
            
            # Create error response
            error_response = {
                "kind": "text",
                "text": f"I encountered an error processing your action: {str(e)}"
            }
            
            # Store error message in database
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
    
    def clear_conversation_buffer(self, conversation_id: str) -> bool:
        """
        Clear the conversation buffer for a specific conversation
        """
        try:
            return self.checkpointer.clear_buffer(conversation_id)
        except Exception as e:
            print(f"Error clearing conversation buffer: {e}")
            return False
    
    def get_buffer_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get buffer information for a conversation
        """
        try:
            return self.checkpointer.get_buffer_info(conversation_id)
        except Exception as e:
            print(f"Error getting buffer info: {e}")
            return None
    
    def trim_conversation_buffer(self, conversation_id: str, max_size: int = 30) -> bool:
        """
        Trim the conversation buffer to a maximum size
        """
        try:
            return self.checkpointer.trim_buffer(conversation_id, max_size)
        except Exception as e:
            print(f"Error trimming conversation buffer: {e}")
            return False