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


class AgentState(Dict):
    """Simplified state for the LangGraph workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: str
    user_id: str
    workflow_status: WorkflowStatus
    context: Dict[str, Any]
    conversation_history: List[BaseMessage]


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
        """Build the simplified LangGraph workflow."""

        workflow = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────
        workflow.add_node("load_buffer", self._load_conversation_buffer)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("finalize", self._finalize_response)  # Combined processing + finalization
        workflow.add_node("save_buffer", self._save_conversation_buffer)

        # ── Entry ──────────────────────────────────────────────
        workflow.set_entry_point("load_buffer")

        # ── Edges ──────────────────────────────────────────────
        workflow.add_edge("load_buffer", "agent")
        workflow.add_edge("tools", "finalize")  # Tools go directly to finalize
        workflow.add_edge("finalize", "save_buffer")
        workflow.add_edge("save_buffer", END)

        # Agent decides: call tools or finish
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": "finalize",  # No tools needed, go straight to finalize
            },
        )

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
            
        except Exception as e:
            print(f"Error loading conversation buffer: {e}")
            state["conversation_history"] = []

        return state

    def _prepare_messages_for_llm(self, state: AgentState) -> List[BaseMessage]:
        """
        Prepare messages for the LLM. We include at most one ToolMessage per 'kind',
        but we always send the very latest HumanMessage—even if it exactly matches history.
        """

        # 1) Check if the last AIMessage requested tools
        last_ai = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        include_tools = bool(last_ai and getattr(last_ai, "tool_calls", None))

        # 2) Start by deduping the buffered ToolMessage objects (keep only the newest of each kind)
        buffer_copy = state["conversation_history"].copy()
        filtered_buffer: List[BaseMessage] = []
        seen_tool_kinds = set()

        for msg in buffer_copy:
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    kind = payload.get("kind", "_generic_tool_")
                except:
                    kind = "_generic_tool_"

                if include_tools:
                    if kind not in seen_tool_kinds:
                        filtered_buffer.append(msg)
                        seen_tool_kinds.add(kind)
                else:
                    # If the last AI had no tool_calls, skip any old ToolMessage entirely
                    continue
            else:
                # Always keep text (Human/AI) in the filtered buffer
                filtered_buffer.append(msg)

        # 3) Now merge in the brand-new turn from state["messages"], but:
        #    • Always append the new HumanMessage/AIMessage (no dedupe on identical text).
        #    • For new ToolMessage objects, only append if we haven't already added that kind.
        merged: List[BaseMessage] = filtered_buffer.copy()
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                # ← Note: We NO LONGER check "if identical content already exists."
                # We always append the new user/assistant message so Gemini sees it.
                merged.append(msg)

            elif isinstance(msg, ToolMessage):
                if not include_tools:
                    continue
                try:
                    payload = json.loads(msg.content)
                    kind = payload.get("kind", "_generic_tool_")
                except:
                    kind = "_generic_tool_"

                if kind not in seen_tool_kinds:
                    merged.append(msg)
                    seen_tool_kinds.add(kind)
                # If the LLM is re-requesting a widget of the same kind, we skip it
                # because we already have the fresh copy.

        # 4) Trim to the last 20 messages
        if len(merged) > 20:
            merged = merged[-20:]

        # 5) Drop any empty content
        final_payload = [m for m in merged if getattr(m, "content", "").strip()]

        print(f"🔍 _prepare_messages_for_llm → sending {len(final_payload)} messages to LLM")
        for i, m in enumerate(final_payload):
            snippet = m.content[:60]
            ell = "…" if len(getattr(m, "content","")) > 60 else ""
            print(f"   [{i:2d}] {type(m).__name__}: {repr(snippet+ell)}")
        return final_payload

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
                print(f"  [{idx:2d}] ({type(m).__name__}): {repr(content[:100])}  length={len(content or '')}")
            
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

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if we should continue to tools or end.
        Enhanced to be more explicit about tool usage decisions.
        """
        
        messages = state["messages"]
        if not messages:
            return "end"
            
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"🔧 Agent requested {len(last_message.tool_calls)} tool(s), continuing to tools")
            return "tools"
        
        print(f"💬 Agent provided text response, ending workflow")
        return "end"
    
    def _extract_final_response_from_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Simplified version: Always use the most recent message content
        """
        
        if not messages:
            return {"kind": "text", "text": "Task completed."}
        
        # Start from the most recent message and work backwards
        for message in reversed(messages):
            
            # Check for tool messages (highest priority)
            if isinstance(message, ToolMessage) and message.content.strip():
                try:
                    # Try parsing as JSON for UI tools
                    content = json.loads(message.content)
                    if content.get("kind") in ["buttons", "inputs", "file_card", "upload_button"]:
                        return content
                except:
                    pass
                
                # Return tool message as text (for calculations, etc.)
                return {
                    "kind": "text",
                    "text": message.content
                }
            
            # Check for AI messages without tool calls
            elif isinstance(message, AIMessage) and message.content.strip():
                # Skip AI messages that have tool calls (they're "thinking" messages)
                if not (hasattr(message, 'tool_calls') and message.tool_calls):
                    return {
                        "kind": "text",
                        "text": message.content
                    }
        
        return {"kind": "text", "text": "Task completed."}

    async def _finalize_response(self, state: AgentState) -> AgentState:
        """
        Simplified finalization that combines tool result processing and response preparation.
        """
        
        messages = state["messages"]
        
        # Extract the final response using single-pass logic
        final_response = self._extract_final_response_from_messages(messages)
        
        # Store in state context
        state["context"]["final_response"] = final_response
        state["workflow_status"] = WorkflowStatus.COMPLETED
        
        print(f"Finalized response: {final_response.get('kind')} - {str(final_response)[:100]}...")
        
        return state
    
    async def _save_conversation_buffer(self, state: AgentState) -> AgentState:
        """Save or update conversation buffer (deduping tool messages by kind)."""

        try:
            conversation_id = state["conversation_id"]

            # 1) Start with the previous buffer from state
            buffer_messages = []
            seen_tool_kinds = set()
            for msg in state["conversation_history"]:
                if isinstance(msg, ToolMessage):
                    try:
                        data = json.loads(msg.content)
                        kind = data.get("kind", "_generic_tool_")
                    except:
                        kind = "_generic_tool_"
                    # Keep only the last occurrence (so skip older if we've seen this kind)
                    if kind not in seen_tool_kinds:
                        buffer_messages.append(msg)
                        seen_tool_kinds.add(kind)
                else:
                    buffer_messages.append(msg)

            # 2) Now merge new messages from state["messages"]
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage):
                    try:
                        data = json.loads(msg.content)
                        kind = data.get("kind", "_generic_tool_")
                    except:
                        kind = "_generic_tool_"
                    if kind not in seen_tool_kinds:
                        buffer_messages.append(msg)
                        seen_tool_kinds.add(kind)
                else:
                    content = getattr(msg, "content", "").strip()
                    if content and not any(
                        getattr(existing, "content", "").strip() == content
                        for existing in buffer_messages
                        if not isinstance(existing, ToolMessage)
                    ):
                        buffer_messages.append(msg)

            # 3) Trim to the last 50 messages
            if len(buffer_messages) > 50:
                buffer_messages = buffer_messages[-50:]

            # 4) Persist to Supabase
            success = self.checkpointer.save_buffer(conversation_id, buffer_messages)
            if success:
                print(f"✅ Successfully saved {len(buffer_messages)} messages to buffer")
                state["conversation_history"] = buffer_messages
            else:
                print("❌ Failed to save buffer")

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
            workflow_status=WorkflowStatus.PENDING,
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
            
            # If no final response was set, use fallback extraction
            if not final_response:
                final_response = self._extract_final_response_from_messages(final_state["messages"])
            
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