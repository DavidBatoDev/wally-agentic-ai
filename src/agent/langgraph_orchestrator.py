# backend/src/agent/langgraph_orchestrator.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.agent_state import AgentState, WorkflowStatus
from src.agent.agent_tools import get_tools
from src.db.checkpointer import (
    _checkpoint_data_to_agent_state,
    load_agent_state,
    save_agent_state,
)
from src.db.db_client import SupabaseClient

log = logging.getLogger(__name__)


class LangGraphOrchestrator:
    """
    LangGraph orchestrator for a document-translation workflow with explicit
    Supabase persistence (load ➜ run ➜ save on every turn).
    """

    # ───────────────────────────────────────── constructor
    def __init__(
        self,
        llm,
        tools: Optional[List[BaseTool]] = None,
        db_client: Optional[SupabaseClient] = None,
    ) -> None:
        if not db_client:
            raise ValueError("db_client is required for persistence")

        self.llm = llm
        self.db_client = db_client
        self.tools: List[BaseTool] = tools or get_tools(db_client=db_client)

        # Bind tools to the LLM and wrap them in a ToolNode runner
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self._tool_runner = ToolNode(self.tools)

        # Pre-compile the graph (no checkpointing at this layer)
        self.graph = self._build_workflow_graph()

    # ───────────────────────────────────────── graph builder
    def _build_workflow_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tool_node)
        g.add_node("finalize", self._finalize_response)

        g.set_entry_point("agent")
        g.add_edge("tools", "finalize")
        g.add_edge("finalize", END)

        g.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": "finalize"},
        )

        return g.compile()

    # ───────────────────────────────────────── prompt helper
    def _create_system_message(self, state: AgentState) -> str:
        """Return a dynamic system prompt describing the current workflow step."""
        base = (
            "You are a specialised document-translation assistant that guides the "
            "user through a structured flow: upload ➜ analyse ➜ extract ➜ translate "
            "➜ template-fill.\n\n"
        )

        ctx: List[str] = []
        if state.user_upload_id and not state.template_id:
            ctx.append("A file has been uploaded. Analyse its type, variation, and language.")
        elif state.template_id and not state.filled_required_fields:
            ctx.append("Template identified. Extract required fields from the upload.")
        elif state.filled_required_fields and state.missing_required_fields:
            ctx.append("Some fields are missing. Ask the user for them.")
        elif (
            state.filled_required_fields
            and not state.missing_required_fields
            and not state.translated_required_fields
        ):
            ctx.append("All fields filled. Translate them next.")
        elif state.translated_required_fields and not state.document_version_id:
            ctx.append("Fields translated. Generate the final document.")
        elif state.document_version_id:
            ctx.append("Translated document generated. Offer download or further help.")
        else:
            ctx.append("New workflow. Ask for document type and target language, then prompt for upload.")

        if state.translate_to:
            ctx.append(f"Target language: {state.translate_to}.")
        if state.template_required_fields:
            keys = list(state.template_required_fields)[:5]
            more = "..." if len(state.template_required_fields) > 5 else ""
            ctx.append(f"Template needs: {', '.join(keys)}{more}.")
        if state.steps_done:
            ctx.append(f"Completed: {', '.join(state.steps_done)}.")

        guidelines = (
            "GUIDELINES:\n"
            "- Use tool calls for UI elements (buttons, forms, uploads).\n"
            "- Never repeat steps already in steps_done.\n"
            "- Provide clear, professional updates.\n"
            "- Reason internally; do not expose chain-of-thought."
        )
        return base + " ".join(ctx) + "\n\n" + guidelines

    # ───────────────────────────────────────── LangGraph nodes
    async def _agent_node(self, state: AgentState) -> AgentState:
        if "agent" not in state.steps_done:
            state.steps_done.append("agent")

        sys_msg = SystemMessage(content=self._create_system_message(state))
        llm_messages: List[BaseMessage] = [sys_msg, *state.messages]

        try:
            response = await self.llm_with_tools.ainvoke(llm_messages)
            state.messages.append(response)
            state.workflow_status = WorkflowStatus.IN_PROGRESS
        except Exception as exc:
            log.exception("_agent_node error")
            state.messages.append(AIMessage(content=f"I encountered an error: {exc}"))
            state.workflow_status = WorkflowStatus.FAILED
        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        if "tools" not in state.steps_done:
            state.steps_done.append("tools")

        last = state.messages[-1] if state.messages else None
        if not getattr(last, "tool_calls", None):
            return state  # No requested tools

        prior = list(state.messages)  # snapshot

        try:
            raw = await self._tool_runner.ainvoke(state)
        except Exception as exc:
            log.exception("tool execution failed")
            state.messages.append(
                ToolMessage(content=f"Tool execution failed: {exc}", tool_call_id="error")
            )
            return state

        merged: AgentState = (
            raw if isinstance(raw, AgentState) else _checkpoint_data_to_agent_state(dict(raw))
        )

        # de-duplicate
        seen = {id(m) for m in prior}
        delta = [m for m in merged.messages or [] if id(m) not in seen]
        merged.messages = prior + delta
        return merged

    def _should_continue(self, state: AgentState) -> str:
        if state.messages and getattr(state.messages[-1], "tool_calls", None):
            return "tools"
        return "end"

    def _extract_final_response(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Pick the most suitable message to surface to the UI."""
        if not messages:
            return {"kind": "text", "text": "Task completed."}

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    if data.get("kind") in {
                        "buttons",
                        "inputs",
                        "file_card",
                        "upload_button",
                    }:
                        return data
                except json.JSONDecodeError:
                    pass
                return {"kind": "text", "text": msg.content}

            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                return {"kind": "text", "text": msg.content}

        return {"kind": "text", "text": "Task completed."}

    async def _finalize_response(self, state: AgentState) -> AgentState:
        if "finalize" not in state.steps_done:
            state.steps_done.append("finalize")

        state.context["final_response"] = self._extract_final_response(state.messages)

        if state.workflow_status != WorkflowStatus.FAILED:
            state.workflow_status = WorkflowStatus.COMPLETED
        return state

    # ───────────────────────────────────────── public API
    async def process_user_message(
        self, conversation_id: str, user_message: str, context: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """Load → run graph → save → return assistant message payload."""
        state = load_agent_state(self.db_client, conversation_id) or AgentState(
            conversation_id=conversation_id,
            user_id=(context or {}).get("user_id", ""),
        )

        state.messages.append(HumanMessage(content=user_message))
        if context:
            state.context.update(context)

        raw = await self.graph.ainvoke(state, RunnableConfig())
        final: AgentState = (
            raw if isinstance(raw, AgentState) else _checkpoint_data_to_agent_state(dict(raw))
        )

        save_agent_state(self.db_client, conversation_id, final)

        final_resp = final.context.get("final_response") or self._extract_final_response(
            final.messages
        )

        assistant_msg = self.db_client.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            kind=final_resp["kind"],
            body=json.dumps(final_resp),
        )

        return {
            "message": assistant_msg,
            "response": final_resp,
            "workflow_status": final.workflow_status.value,
            "steps_completed": len(final.steps_done),
        }

    # ------------------------------------------------------------------ UI action wrapper
    async def handle_user_action(
        self, conversation_id: str, action: str, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert button clicks / uploads / form submissions into natural-language
        messages so the main orchestrator can process them.
        """
        log.debug("handle_user_action action=%s values=%s", action, values)

        try:
            if action.startswith("button_") or action in {
                "yes",
                "no",
                "confirm",
                "cancel",
            }:
                text = (
                    values.get("label")
                    or values.get("value")
                    or values.get("text")
                    or action.replace("button_", "").replace("_", " ").title()
                )
                msg = f"My answer is: {text}"
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "button_context": {
                        "selected_text": text,
                        "selected_action": action,
                        "is_answer": True,
                    },
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            if action == "file_uploaded":
                info = values.get("file_info", {})
                msg = f"I uploaded a file: {info.get('name', 'unknown')}"
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "file_info": info,
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            if action == "form_submitted":
                form_data = values.get("form_data", {})
                parts = ", ".join(f"{k}: {v}" for k, v in form_data.items())
                msg = f"I submitted the form with: {parts}"
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "form_data": form_data,
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            # Fallback
            fallback = values.get("message", f"I selected: {action}")
            return await self.process_user_message(conversation_id, fallback, values)

        except Exception as exc:
            log.exception("handle_user_action failed")
            error = {"kind": "text", "text": f"I encountered an error: {exc}"}
            assistant_msg = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps(error),
            )
            return {
                "message": assistant_msg,
                "response": error,
                "workflow_status": WorkflowStatus.FAILED.value,
                "steps_completed": 0,
            }

    # ------------------------------------------------------------------ helpers
    def get_workflow_status(self, conversation_id: str) -> Optional[str]:
        state = load_agent_state(self.db_client, conversation_id)
        return state.workflow_status.value if state else None

    def clear_conversation_state(self, conversation_id: str) -> bool:
        try:
            (
                self.db_client.client.table("agent_state")
                .delete()
                .eq("conversation_id", conversation_id)
                .execute()
            )
            return True
        except Exception:
            log.exception("clear_conversation_state failed")
            return False
