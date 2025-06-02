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
from agent.core_tools import get_tools
from src.db.checkpointer import (
    _checkpoint_data_to_agent_state,
    load_agent_state,
    save_agent_state,
)
from src.db.db_client import SupabaseClient, supabase_client
from src.agent.analyze_tools import get_analyze_tool
from src.agent.agent_state import UploadAnalysis


log = logging.getLogger(__name__)


class LangGraphOrchestrator:
    """
    LangGraph orchestrator for a document-translation workflow with explicit
    Supabase persistence (load ➜ run ➜ save on every turn).
    """

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

        # Rename: use core_tools instead of tools
        self.core_tools: List[BaseTool] = tools or get_tools(db_client=db_client)
        self.analyze_tool = get_analyze_tool(llm)  # Pass LLM to analyze tool

        # Bind core_tools to the LLM and wrap them in a ToolNode runner
        self.llm_with_core_tools = self.llm.bind_tools(self.core_tools)
        self._analyze_runner = ToolNode([self.analyze_tool])
        self._core_tool_runner = ToolNode(self.core_tools)

        # Pre-compile the graph (no checkpointing at this layer)
        self.graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph for the document-translation workflow.

        ✔  Implemented nodes            → active
        🕐  Future nodes / placeholders → commented-out
        """
        g = StateGraph(AgentState)

        # ── 1. Active nodes ────────────────────────────────────────────────
        g.add_node("load_state_graph", self._load_state_graph)
        g.add_node("agent",            self._agent_node)
        g.add_node("analyze_doc",      self._analyze_doc_node)
        g.add_node("tools",            self._tool_node)
        g.add_node("save_state_graph", self._save_state_graph)
        g.add_node("finalize",         self._finalize_response)

        # ── 2. Place-holders (add real callables later) ───────────────────
        # g.add_node("find_template",             self._find_template_node)          # 🕐
        # g.add_node("extract_required_fields",   self._extract_required_fields_node) # 🕐
        # g.add_node("ask_node",                  self._ask_node)                    # 🕐
        # g.add_node("translate_required_fields", self._translate_required_fields_node) # 🕐
        # g.add_node("fill_template",             self._fill_template_node)          # 🕐
        # g.add_node("change_specific_field",     self._change_specific_field_node)  # 🕐
        # g.add_node("manual_fill_template",      self._manual_fill_template_node)   # 🕐
        # g.add_node("end_node",                  self._end_node)                    # 🕐

        # ── 3. Linear backbone you can already run ────────────────────────
        g.set_entry_point("load_state_graph")
        g.add_edge("load_state_graph", "agent")

        # First upload → analyse → save
        g.add_edge("analyze_doc",      "save_state_graph")

        # Generic tool-invocation → save
        g.add_edge("tools",            "save_state_graph")

        # Always persist then finalise
        g.add_edge("save_state_graph", "finalize")
        g.add_edge("finalize",         END)

        # ── 4. Dynamic router after *every* agent turn ─────────────────────
        g.add_conditional_edges(
            "agent",
            self._route_next,                # your smart router
            {
                "analyze_doc": "analyze_doc",
                # "find_template":             "find_template",            # 🕐
                # "extract_required_fields":   "extract_required_fields",  # 🕐
                # "ask":                       "ask_node",                 # 🕐
                # "translate":                 "translate_required_fields",# 🕐
                # "fill_template":             "fill_template",            # 🕐
                # "manual_fill":               "manual_fill_template",     # 🕐
                # "change_field":              "change_specific_field",    # 🕐
                # "end":                       "end_node",                 # 🕐
                "tools":       "tools",
                "save":        "save_state_graph",
            },
        )

        return g.compile()

    def _create_system_message(self, state: AgentState) -> str:
        """Return a dynamic system prompt describing the current workflow step."""
        base = (
            "You are a specialised document-translation assistant that guides the "
            "user through a structured flow: upload ➜ analyse ➜ extract ➜ translate "
            "➜ template-fill.\n\n"
        )

        ctx: List[str] = []
        if state.user_upload_id and not state.upload_analysis:
            ctx.append("A file has been uploaded. Analyse its type, variation, and language.")
        elif state.upload_analysis and not state.template_id:
            ctx.append("Document analyzed. Find matching template based on analysis.")
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
        if state.upload_analysis:
            analysis = state.upload_analysis
            ctx.append(f"Document analyzed: {analysis.get('doc_type', 'unknown')} ({analysis.get('variation', 'unknown')}) in {analysis.get('detected_language', 'unknown')}.")
        if state.template_required_fields:
            keys = list(state.template_required_fields)[:5]
            more = "..." if len(state.template_required_fields) > 5 else ""
            ctx.append(f"Template needs: {', '.join(keys)}{more}.")
        if state.steps_done:
            ctx.append(f"Completed: {', '.join(state.steps_done)}.")

        guidelines = (
            "GUIDELINES:\n"
            "- Use tool calls for UI elements (buttons, forms, uploads).\n"
            "- You can call MULTIPLE tools in a single response when needed.\n"
            "- For example: call update_agent_state AND show_buttons together.\n"
            "- Never repeat steps already in steps_done.\n"
            "- Provide clear, professional updates.\n"
            "- Reason internally; do not expose chain-of-thought.\n"
            f"- ALWAYS include conversation_id='{state.conversation_id}' when calling tools that support it.\n"
        )

        tool_hint = (
            "\n\nIf the user message gives you ANY new detail about the workflow "
            "(translate_to, translate_from  etc.) **call the tool "
            "`update_agent_state`** with only the keys that changed. "
            "You can then ALSO call other tools like show_buttons or show_upload_button "
            "in the same response to provide the next UI interaction. "
            f"Remember to always pass conversation_id='{state.conversation_id}' to tools. "
            "After all tool calls, send your normal assistant reply."
        )

        return (
            base
            + f"Current state: [{self._state_summary(state)}]. "
            + " ".join(ctx)
            + "\n\n"
            + guidelines
            + tool_hint
        )

    async def _agent_node(self, state: AgentState) -> AgentState:
        # Mark that _agent_node has run
        if "agent" not in state.steps_done:
            state.steps_done.append("agent")

        # Build a SystemMessage + existing chat history so far
        sys_msg = SystemMessage(content=self._create_system_message(state))
        llm_messages: List[BaseMessage] = [sys_msg, *state.messages]

        # Invoke the LLM (which may emit one or more tool_calls)
        response: AIMessage = await self.llm_with_core_tools.ainvoke(llm_messages)

        # If LLM produced any tool_calls named "update_agent_state", strip out conversation_id
        # and apply the remaining patch keys to our in-memory state.
        for tc in getattr(response, "tool_calls", []):
            if tc["name"] == "update_agent_state":
                patch = {k: v for k, v in tc["args"].items() if k != "conversation_id"}
                self._apply_state_patch(state, patch)

        state.messages.append(response)
        state.workflow_status = WorkflowStatus.IN_PROGRESS
        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        # Mark that we're running tool-invocation
        if "tools" not in state.steps_done:
            state.steps_done.append("tools")

        last = state.messages[-1]
        # If that last message had no tool_calls, skip running tools
        if not getattr(last, "tool_calls", None):
            return state

        prior = list(state.messages)

        # Make sure the `conversation_id` is in state.context so every tool sees it
        tool_context = {
            "conversation_id": state.conversation_id,
            "user_id": getattr(state, "user_id", ""),
        }
        state.context.update(tool_context)

        # Now invoke the tools. LangGraph will look at state.context + the tool_calls
        # objects in the last AIMessage, match by name, and run them one by one.
        raw = await self._core_tool_runner.ainvoke(state)

        merged = state.model_copy(deep=True)
        new_msgs: List[BaseMessage] = []
        if isinstance(raw, AgentState):
            new_msgs = raw.messages
        else:
            # raw is typically an AddableValuesDict with a "messages" list
            new_msgs = list(raw.get("messages", []))

        # De-duplicate: only append those new messages which are not already in `prior`
        seen_ids = {id(m) for m in prior}
        merged.messages = prior + [m for m in new_msgs if id(m) not in seen_ids]
        return merged
    
    async def _analyze_doc_node(self, state: AgentState) -> AgentState:
        """
        Use the analyze_upload tool to analyze the uploaded document.
        The tool returns analysis data directly which we store in state.upload_analysis.
        """
        if "analyze_doc" in state.steps_done:
            # Already done in a previous turn
            return state

        # Insert a message that we're analyzing the document
        analysis_msg = AIMessage(content="🔍 Analyzing the uploaded document to determine its type and language...")
        state.messages.append(analysis_msg)
        
        # Create message in database too
        self.db_client.create_message(
            conversation_id=state.conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps({
                "text": "🔍 Analyzing the uploaded document to determine its type and language..."
            }),
        )

        # Mark this step as done
        state.steps_done.append("analyze_doc")

        try:
            # Use the analyze tool to get analysis
            analysis_result = self.analyze_tool.invoke({
                "file_id": state.user_upload_id,
                "public_url": state.user_upload_public_url
            })
            
            # Store the analysis in state
            state.upload_analysis = UploadAnalysis(**analysis_result)
            
            # Create a user-friendly summary
            doc_type = analysis_result.get("doc_type", "unknown")
            variation = analysis_result.get("variation", "standard")
            language = analysis_result.get("detected_language", "unknown")
            confidence = analysis_result.get("confidence", 0.0)
            
            confidence_text = ""
            if confidence > 0.8:
                confidence_text = " (high confidence)"
            elif confidence > 0.5:
                confidence_text = " (medium confidence)"
            elif confidence > 0.0:
                confidence_text = " (low confidence)"
            
            summary_parts = [
                f"✅ **Document Analysis Complete**",
                f"• **Type**: {doc_type.replace('_', ' ').title()}",
                f"• **Variation**: {variation.replace('_', ' ').title()}",
                f"• **Language**: {language.replace('_', ' ').title()}{confidence_text}",
            ]
            
            if analysis_result.get("page_count"):
                summary_parts.append(f"• **Pages**: {analysis_result['page_count']}")
            if analysis_result.get("page_size"):
                summary_parts.append(f"• **Size**: {analysis_result['page_size']}")
            
            summary_msg = AIMessage(content="\n".join(summary_parts))
            state.messages.append(summary_msg)
            
            # Also update the detected language in state if we found one
            if language and language != "unknown" and not state.translate_from:
                state.translate_from = language
                
        except Exception as e:
            log.exception(f"Error analyzing document: {e}")
            
            # Store minimal analysis on error
            state.upload_analysis = {
                "file_id": state.user_upload_id,
                "doc_type": "unknown",
                "variation": "standard", 
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
            
            error_msg = AIMessage(content="⚠️ Unable to fully analyze the document. Please verify the document type manually.")
            state.messages.append(error_msg)

        return state

    async def _load_state_graph(self, state: AgentState) -> AgentState:
        db_state = load_agent_state(self.db_client, state.conversation_id)
        if db_state:
            # Merge "new" messages & context from this turn into the persisted baseline
            delta = state.messages[len(db_state.messages) :]
            db_state.messages.extend(delta)
            db_state.context.update(state.context)
            return db_state

        # No existing row → just continue with the fresh state we were given
        return state

    async def _save_state_graph(self, state: AgentState) -> AgentState:
        try:
            save_agent_state(self.db_client, state.conversation_id, state)
        except Exception as exc:
            log.exception("save_state_graph failed: %s", exc)
        return state

    def _route_next(self, state: AgentState) -> str:
        """
        Decide what the next node should be.

        Order of precedence
        -------------------
        1.  If the user has just uploaded a file and we have not analysed it yet
            → jump to the `analyze_doc` node.
        2.  If the last AIMessage contains tool calls
            → jump to the generic `tools` node.
        3.  Otherwise
            → go straight to `save_state_graph`.
        """
        if state.user_upload_id and "analyze_doc" not in state.steps_done:
            return "analyze_doc"
        
        if state.messages and getattr(state.messages[-1], "tool_calls", None):
            return "tools"
        
        return "save"

    def _extract_final_response(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Pick the most suitable message to surface to the UI."""
        if not messages:
            return {"kind": "text", "text": "Task completed."}

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    # If this tool message is a UI payload (buttons, upload, file_card),
                    # return it directly as the final response.
                    if data.get("kind") in {
                        "buttons",
                        "inputs",
                        "file_card",
                        "upload_button",
                    }:
                        return data
                except json.JSONDecodeError:
                    pass
                # Otherwise return the raw content string
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

    async def process_user_message(
        self, conversation_id: str, user_message: str, context: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """
        Public entry point for a new user message.
        1. Load any existing state from Supabase (or start fresh).
        2. Append the new HumanMessage.
        3. Run the graph: load ➜ agent ➜ (tools?) ➜ save ➜ finalize.
        4. Persist the updated AgentState back to Supabase.
        5. Return the "final_response" (whatever UI payload or text the agent emitted).
        """
        # 1.  Load or initialize state
        state = load_agent_state(self.db_client, conversation_id) or AgentState(
            conversation_id=conversation_id,
            user_id=(context or {}).get("user_id", ""),
        )

        # 2.  Append the incoming user message to the in-memory state
        state.messages.append(HumanMessage(content=user_message))

        # 3.  Merge any additional context (e.g. user_id, source_action, etc.)
        if context:
            state.context.update(context)

        # 4.  Run the state-graph
        raw = await self.graph.ainvoke(state, RunnableConfig())
        final: AgentState = (
            raw
            if isinstance(raw, AgentState)
            else _checkpoint_data_to_agent_state(dict(raw))
        )

        # 5.  Persist the final state to Supabase
        save_agent_state(self.db_client, conversation_id, final)

        # 6.  Figure out what "final_response" to send to the UI
        final_resp = final.context.get("final_response") or self._extract_final_response(
            final.messages
        )

        # 7.  Insert that final payload into the messages table,
        #     so that the real-time client sees it
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
    
    async def process_user_file(
        self, conversation_id: str, file_info: Dict[str, Any], context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        Public entry point for a file upload.
        Creates a human message describing the file and processes it through the workflow.
        """
        # 1. Load or initialize state
        state = load_agent_state(self.db_client, conversation_id) or AgentState(
            conversation_id=conversation_id,
            user_id=(context or {}).get("user_id", ""),
        )
        
        # 2. Create a natural language message about the file upload
        file_message = (
            f"[FILE_UPLOADED] I uploaded **{file_info.get('filename', 'a file')}** "
            f"({file_info.get('mime_type', 'unknown type')}, "
            f"{file_info.get('size_bytes', 0):,} bytes). "
            f"You can access it at: {file_info.get('public_url', '')}"
        )
        
        # 3. Append the file info as a HumanMessage
        state.messages.append(HumanMessage(content=file_message))

        # 4. Merge any additional context
        if context:
            state.context.update(context)
        
        # 5. Store file info directly in state
        state.user_upload_id = file_info.get("file_id", "")
        state.user_upload_public_url = file_info.get("public_url", "")
        
        # Also add file info to context for tools
        state.context.update({
            "uploaded_file": file_info,
            "file_id": file_info.get("file_id"),
            "public_url": file_info.get("public_url"),
        })

        # 6. Run the state-graph
        raw = await self.graph.ainvoke(state, RunnableConfig())
        final: AgentState = (
            raw
            if isinstance(raw, AgentState)
            else _checkpoint_data_to_agent_state(dict(raw))
        )

        # 7. Persist the final state to Supabase
        save_agent_state(self.db_client, conversation_id, final)

        # 8. Figure out what "final_response" to send to the UI
        final_resp = final.context.get("final_response") or self._extract_final_response(
            final.messages
        )

        # 9. Insert that final payload into the messages table
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

    def _state_summary(self, s: AgentState) -> str:
        """Return a terse, 1-line summary for the system prompt."""
        parts = []
        if s.translate_to:
            parts.append(f"translate_to={s.translate_to}")
        if s.translate_from:
            parts.append(f"translate_from={s.translate_from}")
        if s.template_id:
            parts.append(f"template_id={s.template_id}")
        if s.user_upload_id:
            parts.append("file_uploaded✔")
        if s.missing_required_fields:
            parts.append(f"missing={list(s.missing_required_fields)[:3]}…")
        return ", ".join(parts) or "empty"

    def _apply_state_patch(self, state: AgentState, patch: Dict[str, Any]) -> None:
        """
        Apply the partial update coming from `update_agent_state` tool calls.
        Simplified to handle only the explicit fields we want to support.
        """
        for key, val in patch.items():
            if not hasattr(state, key):
                # Silently skip anything not on AgentState
                continue

            # Simple type validation for key fields
            if key in ("user_upload_id", "user_upload_public_url") and not isinstance(val, str):
                log.warning("%s must be str – got %s", key, type(val))
                continue

            # Default behaviour: let Pydantic handle the assignment
            setattr(state, key, val)