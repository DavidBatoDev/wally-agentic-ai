# backend/src/agent/langgraph_orchestrator.py
from typing import Dict, Any, List, Optional, Union, Literal
import json
from dataclasses import dataclass
from enum import Enum

from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage
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
        
        # Create tool node for LangGraph
        self.tool_node = ToolNode(self.tools)
        
        # Initialize memory for state persistence
        self.memory = MemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the state schema
        def create_initial_state() -> AgentState:
            return AgentState(
                messages=[],
                conversation_id="",
                user_id="",
                current_step=None,
                workflow_steps=[],
                workflow_status=WorkflowStatus.PENDING,
                intermediate_results={},
                user_confirmation_pending=False,
                context={}
            )
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("plan_workflow", self._plan_workflow)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("request_confirmation", self._request_confirmation)
        workflow.add_node("handle_confirmation", self._handle_confirmation)
        workflow.add_node("finalize_response", self._finalize_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("analyze_request")
        
        # Add edges with conditional logic
        workflow.add_edge("analyze_request", "plan_workflow")
        
        workflow.add_conditional_edges(
            "plan_workflow",
            self._should_execute_or_confirm,
            {
                "execute": "execute_step",
                "request_confirmation": "request_confirmation",
                "finalize": "finalize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_step",
            self._after_step_execution,
            {
                "continue": "execute_step",
                "request_confirmation": "request_confirmation",
                "finalize": "finalize_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("request_confirmation", "handle_confirmation")
        
        workflow.add_conditional_edges(
            "handle_confirmation",
            self._after_confirmation,
            {
                "continue": "execute_step",
                "finalize": "finalize_response",
                "cancel": END
            }
        )
        
        workflow.add_edge("finalize_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _analyze_request(self, state: AgentState) -> AgentState:
        """Analyze the user request to understand intent and complexity"""
        
        latest_message = state["messages"][-1] if state["messages"] else None
        if not latest_message:
            state["workflow_status"] = WorkflowStatus.FAILED
            return state
        
        user_input = latest_message.content
        
        # Use LLM to analyze the request
        analysis_prompt = f"""
        Analyze this user request and determine:
        1. Is this a simple single-step task or a complex multi-step workflow?
        2. What are the main components/steps needed?
        3. Does any step require user confirmation?
        
        User request: {user_input}
        
        Respond in JSON format:
        {{
            "complexity": "simple" | "multi_step",
            "estimated_steps": number,
            "requires_confirmation": boolean,
            "intent": "brief description",
            "suggested_steps": ["step1", "step2", ...]
        }}
        """
        
        analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
        
        try:
            analysis = json.loads(analysis_response.content)
            state["context"]["request_analysis"] = analysis
            state["workflow_status"] = WorkflowStatus.IN_PROGRESS
        except json.JSONDecodeError:
            state["context"]["request_analysis"] = {
                "complexity": "simple",
                "estimated_steps": 1,
                "requires_confirmation": False,
                "intent": "Process user request",
                "suggested_steps": ["execute_request"]
            }
        
        return state
    
    async def _plan_workflow(self, state: AgentState) -> AgentState:
        """Plan the workflow steps based on request analysis"""
        
        analysis = state["context"].get("request_analysis", {})
        
        if analysis.get("complexity") == "simple":
            # Simple single-step workflow
            step = WorkflowStep(
                id="single_step",
                name="Execute Request",
                description="Process the user request directly",
                requires_confirmation=False
            )
            state["workflow_steps"] = [step]
        else:
            # Multi-step workflow
            suggested_steps = analysis.get("suggested_steps", ["step1"])
            workflow_steps = []
            
            for i, step_name in enumerate(suggested_steps):
                step = WorkflowStep(
                    id=f"step_{i+1}",
                    name=step_name.title(),
                    description=f"Execute {step_name}",
                    requires_confirmation=(i > 0 and analysis.get("requires_confirmation", False))
                )
                workflow_steps.append(step)
            
            state["workflow_steps"] = workflow_steps
        
        state["current_step"] = state["workflow_steps"][0].id if state["workflow_steps"] else None
        return state
    
    async def _execute_step(self, state: AgentState) -> AgentState:
        """Execute the current workflow step"""
        
        if not state["current_step"]:
            state["workflow_status"] = WorkflowStatus.COMPLETED
            return state
        
        current_step = next(
            (step for step in state["workflow_steps"] if step.id == state["current_step"]),
            None
        )
        
        if not current_step:
            state["workflow_status"] = WorkflowStatus.FAILED
            return state
        
        try:
            # Mark step as in progress
            current_step.status = WorkflowStatus.IN_PROGRESS
            
            # Get the latest user message for context
            user_message = next(
                (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
                ""
            )
            
            # Create execution prompt
            execution_prompt = f"""
            Execute this workflow step:
            Step: {current_step.name}
            Description: {current_step.description}
            User Request: {user_message}
            Previous Results: {json.dumps(state["intermediate_results"], indent=2)}
            
            Available tools: {[tool.name for tool in self.tools]}
            
            Execute the step and provide results. Use tools if necessary.
            """
            
            # Execute with tools
            response = await self.llm.ainvoke([HumanMessage(content=execution_prompt)])
            
            # Check if LLM wants to use tools
            if "tool_calls" in response.additional_kwargs:
                # Handle tool calls
                tool_calls = response.additional_kwargs["tool_calls"]
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    # Find and execute the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        tool_result = await tool.ainvoke(tool_args)
                        current_step.result = {
                            "tool_used": tool_name,
                            "result": tool_result
                        }
            else:
                # Regular LLM response
                current_step.result = {
                    "response": response.content
                }
            
            current_step.status = WorkflowStatus.COMPLETED
            state["intermediate_results"][current_step.id] = current_step.result
            
        except Exception as e:
            current_step.status = WorkflowStatus.FAILED
            current_step.error = str(e)
            state["workflow_status"] = WorkflowStatus.FAILED
        
        return state
    
    async def _request_confirmation(self, state: AgentState) -> AgentState:
        """Request user confirmation for the next step"""
        
        current_step = next(
            (step for step in state["workflow_steps"] if step.id == state["current_step"]),
            None
        )
        
        if current_step:
            # Create confirmation request
            confirmation_data = {
                "kind": "buttons",
                "prompt": f"Ready to proceed with: {current_step.name}?",
                "buttons": [
                    {"label": "Continue", "action": "confirm_continue"},
                    {"label": "Skip Step", "action": "confirm_skip"},
                    {"label": "Cancel", "action": "confirm_cancel"}
                ]
            }
            
            state["user_confirmation_pending"] = True
            state["context"]["pending_confirmation"] = confirmation_data
        
        return state
    
    async def _handle_confirmation(self, state: AgentState) -> AgentState:
        """Handle user confirmation response"""
        
        # In a real implementation, this would wait for user input
        # For now, we'll assume confirmation is granted
        state["user_confirmation_pending"] = False
        state["context"]["confirmation_result"] = "continue"
        
        return state
    
    async def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize and format the response"""
        
        # Compile all results
        results = []
        for step in state["workflow_steps"]:
            if step.status == WorkflowStatus.COMPLETED and step.result:
                results.append(step.result)
        
        # Create final response
        if len(results) == 1 and "response" in results[0]:
            # Single response
            final_response = {
                "kind": "text",
                "text": results[0]["response"]
            }
        elif any("tool_used" in result for result in results):
            # Tool-based response
            tool_results = [r for r in results if "tool_used" in r]
            if tool_results and isinstance(tool_results[-1]["result"], dict):
                final_response = tool_results[-1]["result"]
            else:
                final_response = {
                    "kind": "text",
                    "text": "Task completed successfully."
                }
        else:
            # Default text response
            final_response = {
                "kind": "text",
                "text": "Workflow completed successfully."
            }
        
        state["context"]["final_response"] = final_response
        state["workflow_status"] = WorkflowStatus.COMPLETED
        
        return state
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle workflow errors"""
        
        error_step = next(
            (step for step in state["workflow_steps"] if step.status == WorkflowStatus.FAILED),
            None
        )
        
        error_message = "An error occurred during workflow execution."
        if error_step and error_step.error:
            error_message = f"Error in {error_step.name}: {error_step.error}"
        
        state["context"]["final_response"] = {
            "kind": "text",
            "text": error_message
        }
        state["workflow_status"] = WorkflowStatus.FAILED
        
        return state
    
    def _should_execute_or_confirm(self, state: AgentState) -> str:
        """Determine whether to execute, request confirmation, or finalize"""
        
        if not state["workflow_steps"]:
            return "finalize"
        
        current_step = next(
            (step for step in state["workflow_steps"] if step.id == state["current_step"]),
            None
        )
        
        if not current_step:
            return "finalize"
        
        if current_step.requires_confirmation and not state.get("user_confirmation_pending"):
            return "request_confirmation"
        
        return "execute"
    
    def _after_step_execution(self, state: AgentState) -> str:
        """Determine next action after step execution"""
        
        current_step = next(
            (step for step in state["workflow_steps"] if step.id == state["current_step"]),
            None
        )
        
        if not current_step:
            return "finalize"
        
        if current_step.status == WorkflowStatus.FAILED:
            return "error"
        
        # Move to next step
        current_index = next(
            (i for i, step in enumerate(state["workflow_steps"]) if step.id == state["current_step"]),
            -1
        )
        
        if current_index >= 0 and current_index < len(state["workflow_steps"]) - 1:
            next_step = state["workflow_steps"][current_index + 1]
            state["current_step"] = next_step.id
            
            if next_step.requires_confirmation:
                return "request_confirmation"
            else:
                return "continue"
        
        return "finalize"
    
    def _after_confirmation(self, state: AgentState) -> str:
        """Determine action after user confirmation"""
        
        confirmation_result = state["context"].get("confirmation_result", "continue")
        
        if confirmation_result == "continue":
            return "continue"
        elif confirmation_result == "cancel":
            return "cancel"
        else:
            return "finalize"
    
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
            final_response = final_state["context"].get("final_response", {
                "kind": "text",
                "text": "I apologize, but I couldn't process your request."
            })
            
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
                "steps_completed": len([s for s in final_state["workflow_steps"] if s.status == WorkflowStatus.COMPLETED])
            }
            
        except Exception as e:
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
        
        config = RunnableConfig(
            configurable={"thread_id": conversation_id}
        )
        
        # Get current state
        current_state = await self.graph.aget_state(config)
        
        if not current_state or not current_state.values.get("user_confirmation_pending"):
            return {
                "error": "No pending confirmation found",
                "response": {"kind": "text", "text": "No action required at this time."}
            }
        
        # Update state with user action
        updated_state = current_state.values.copy()
        updated_state["context"]["confirmation_result"] = action
        updated_state["user_confirmation_pending"] = False
        
        # Continue workflow execution
        final_state = await self.graph.ainvoke(updated_state, config)
        
        # Get the response
        final_response = final_state["context"].get("final_response", {
            "kind": "text",
            "text": "Action processed."
        })
        
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
            "workflow_status": final_state["workflow_status"].value
        }