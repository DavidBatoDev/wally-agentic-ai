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
# from src.agent.analyze_tools import get_analyze_tool
from src.agent.functions.smart_router import SmartRouter
from src.agent.helpers import create_message_if_not_duplicate


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
        use_smart_routing: bool = True,
    ) -> None:
        if not db_client:
            raise ValueError("db_client is required for persistence")

        self.llm = llm
        self.db_client = db_client
        self.use_smart_routing = use_smart_routing

        # Rename: use core_tools instead of tools
        self.core_tools: List[BaseTool] = tools or get_tools(db_client=db_client)

        # Bind core_tools to the LLM and wrap them in a ToolNode runner
        self.llm_with_core_tools = self.llm.bind_tools(self.core_tools)
        self._core_tool_runner = ToolNode(self.core_tools)

        self.smart_router = None
        if self.use_smart_routing:
            try:
                self.smart_router = SmartRouter()
                log.info("Smart router initialized successfully")
            except Exception as e:
                log.warning(f"Failed to initialize smart router: {e}. Falling back to rule-based routing.")
                self.use_smart_routing = False



        # Pre-compile the graph (no checkpointing at this layer)
        self.graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph for the document-translation workflow.

        ✔  Implemented nodes            → active
        🕐  Future nodes / placeholders → now uncommented with print statements
        """
        g = StateGraph(AgentState)

        # ── 1. Active nodes ────────────────────────────────────────────────
        g.add_node("load_state_graph", self._load_state_graph)
        g.add_node("agent",            self._agent_node)
        g.add_node("analyze_doc",      self._analyze_doc_node)
        g.add_node("tools",            self._tool_node)
        g.add_node("save_state_graph", self._save_state_graph)
        g.add_node("finalize",         self._finalize_response)

        # ── 2. Placeholder nodes (now uncommented with debug prints) ───────
        g.add_node("find_template",             self._find_template_node)
        g.add_node("extract_required_fields",   self._extract_required_fields_node)
        g.add_node("translate_required_fields", self._translate_required_fields_node)
        g.add_node("fill_template",             self._fill_template_node)
        g.add_node("change_specific_field",     self._change_specific_field_node)
        g.add_node("manual_fill_template",      self._manual_fill_template_node)
        g.add_node("end_node",                  self._end_node)

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
                "analyze_doc":              "analyze_doc",
                "find_template":             "find_template",
                "extract_required_fields":   "extract_required_fields",
                "translate":                 "translate_required_fields",
                "fill_template":             "fill_template",
                "manual_fill":               "manual_fill_template",
                "change_field":              "change_specific_field",
                "end":                       "end_node",
                "tools":                    "tools",
                "save":                     "save_state_graph",
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
            "(translate_to, translate_from) **call the tool "
            "`update_agent_state`** with only the keys that changed (the only thing that update_agent_state should translate_to and translate_from). "
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
        This calls the analyze tool directly in a deterministic way.
        """
        print(f"🔍 [ANALYZE_DOC] Starting for conversation: {state.conversation_id}")
        
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
            print(f"🔍 [ANALYZE_DOC] Calling analyze tool directly...")

            # import analyze_tool result
            from agent.functions.analyze_doc_node_helper import analyze_upload
            
            if state.user_upload_public_url == "" or state.user_upload_public_url is None:
                raise ValueError("Missing required upload URL for document analysis")

            analysis_result = await analyze_upload(
                file_id=state.user_upload_id,
                public_url=state.user_upload_public_url
            )
            
            # Store the analysis in state
            state.upload_analysis = {**analysis_result}
            
            print(f"🔍 [ANALYZE_DOC] Stored analysis in state")
            
            # Create a user-friendly summary
            doc_type = analysis_result.get("doc_type", "unknown")
            variation = analysis_result.get("variation", "standard")
            language = analysis_result.get("detected_language", "unknown")

            summary_parts = [
                f"✅ **Document Analysis Complete**",
                f"• **Type**: {doc_type.replace('_', ' ').title()}\n",
                f"• **Variation**: {variation.replace('_', ' ').title()}\n",
                f"• **Language**: {language.replace('_', ' ').title()}\n",
            ]
            
            if analysis_result.get("page_count"):
                summary_parts.append(f"• **Pages**: {analysis_result['page_count']}\n")
            if analysis_result.get("page_size"):
                summary_parts.append(f"• **Size**: {analysis_result['page_size']}\n")

            upload_summary = analysis_result.get("content_summary", "")
            summary_parts.append(f"• **Summary**: {upload_summary}\n")
            
            summary_msg = AIMessage(content="\n".join(summary_parts))
            state.messages.append(summary_msg)
            
            # Create message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": "\n".join(summary_parts)
                }),
            )

            state.workflow_status = WorkflowStatus.WAITING_CONFIRMATION
            
            print(f"🔍 [ANALYZE_DOC] Analysis completed successfully")
            
        except Exception as e:
            print(f"🔍 [ANALYZE_DOC] ERROR: {str(e)}")
            log.exception(f"Error analyzing document: {e}")
            
            # Store minimal analysis on error
            state.upload_analysis = {
                "file_id": state.user_upload_id,
                "mime_type": "unknown",
                "doc_type": "unknown",
                "variation": "standard", 
                "detected_language": "unknown",
                "confidence": 0.0
                }
            
            error_msg = AIMessage(content="⚠️ Unable to fully analyze the document. Please verify the document type manually.")
            state.messages.append(error_msg)
            
            # Create error message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": "⚠️ Unable to fully analyze the document. Please verify the document type manually."
                }),
            )

        print(f"🔍 [ANALYZE_DOC] Completed")
        return state



    # ── PLACEHOLDER NODES (now uncommented with debug prints) ──────────────

    async def _find_template_node(self, state: AgentState) -> AgentState:
        """Find matching template based on document analysis."""
        print("🔍 [DEBUG] Executing _find_template_node")
        log.info("find_template_node executed for conversation %s", state.conversation_id)
        
        if "find_template" not in state.steps_done:
            state.steps_done.append("find_template")
        
        # TODO: Implement template finding logic
        # For now, just add a placeholder message
        msg = AIMessage(content="🔍 Finding matching template based on document analysis...")
        state.messages.append(msg)
        
        return state

    async def _extract_required_fields_node(self, state: AgentState) -> AgentState:
        """Extract required fields from the uploaded document."""
        print("📋 [DEBUG] Executing _extract_required_fields_node")
        log.info("extract_required_fields_node executed for conversation %s", state.conversation_id)
        
        if "extract_required_fields" not in state.steps_done:
            state.steps_done.append("extract_required_fields")
        
        # TODO: Implement field extraction logic
        msg = AIMessage(content="📋 Extracting required fields from the document...")
        state.messages.append(msg)
        
        return state


    async def _translate_required_fields_node(self, state: AgentState) -> AgentState:
        """Translate the required fields to target language."""
        print("🌐 [DEBUG] Executing _translate_required_fields_node")
        log.info("translate_required_fields_node executed for conversation %s", state.conversation_id)
        
        if "translate_required_fields" not in state.steps_done:
            state.steps_done.append("translate_required_fields")
        
        # TODO: Implement translation logic
        msg = AIMessage(content="🌐 Translating required fields to target language...")
        state.messages.append(msg)
        
        return state

    async def _fill_template_node(self, state: AgentState) -> AgentState:
        """Fill the template with translated fields."""
        print("📝 [DEBUG] Executing _fill_template_node")
        log.info("fill_template_node executed for conversation %s", state.conversation_id)
        
        if "fill_template" not in state.steps_done:
            state.steps_done.append("fill_template")
        
        # TODO: Implement template filling logic
        msg = AIMessage(content="📝 Filling template with translated information...")
        state.messages.append(msg)
        
        return state

    async def _change_specific_field_node(self, state: AgentState) -> AgentState:
        """Change a specific field in the document."""
        print("✏️ [DEBUG] Executing _change_specific_field_node")
        log.info("change_specific_field_node executed for conversation %s", state.conversation_id)
        
        if "change_specific_field" not in state.steps_done:
            state.steps_done.append("change_specific_field")
        
        # TODO: Implement field changing logic
        msg = AIMessage(content="✏️ Changing specific field in the document...")
        state.messages.append(msg)
        
        return state

    async def _manual_fill_template_node(self, state: AgentState) -> AgentState:
        """Manually fill template with user-provided information."""
        print("✍️ [DEBUG] Executing _manual_fill_template_node")
        log.info("manual_fill_template_node executed for conversation %s", state.conversation_id)
        
        if "manual_fill_template" not in state.steps_done:
            state.steps_done.append("manual_fill_template")
        
        # TODO: Implement manual template filling logic
        msg = AIMessage(content="✍️ Manually filling template with user information...")
        state.messages.append(msg)
        
        return state

    async def _end_node(self, state: AgentState) -> AgentState:
        """End the workflow process."""
        print("🏁 [DEBUG] Executing _end_node")
        log.info("end_node executed for conversation %s", state.conversation_id)
        
        if "end" not in state.steps_done:
            state.steps_done.append("end")
        
        # TODO: Implement workflow completion logic
        msg = AIMessage(content="🏁 Workflow completed successfully!")
        state.messages.append(msg)
        
        state.workflow_status = WorkflowStatus.COMPLETED
        return state

    # ── END PLACEHOLDER NODES ──────────────────────────────────────────────
    def _merge_states(self, current_state: AgentState, db_state: AgentState) -> AgentState:
        """
        Merge current state (with new data) into persisted state (from DB).
        Priority: current_state values override db_state when current has non-empty values.
        """

        print('states in merge_state', current_state.user_upload_public_url)
        # Start with db_state as base
        merged = db_state.model_copy(deep=True)
        
        # Merge messages (only new ones)
        delta = current_state.messages[len(db_state.messages):]
        merged.messages.extend(delta)
        
        # Merge context
        merged.context.update(current_state.context)
        
        # Preserve file upload info from current state if it's newer/different
        if current_state.user_upload_id and current_state.user_upload_id != merged.user_upload_id:
            merged.user_upload_id = current_state.user_upload_id
        if current_state.user_upload_public_url and current_state.user_upload_public_url != merged.user_upload_public_url:
            merged.user_upload_public_url = current_state.user_upload_public_url
        if current_state.upload_analysis and current_state.upload_analysis != merged.upload_analysis:
            merged.upload_analysis = current_state.upload_analysis
        
        # Preserve other potentially new fields from current state
        if current_state.translate_to and current_state.translate_to != merged.translate_to:
            merged.translate_to = current_state.translate_to
        if current_state.translate_from and current_state.translate_from != merged.translate_from:
            merged.translate_from = current_state.translate_from
        
        return merged

    async def _load_state_graph(self, state: AgentState) -> AgentState:
        db_state = load_agent_state(self.db_client, state.conversation_id)
        if db_state:
            merged_state = self._merge_states(state, db_state)
            return merged_state

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
        Decide what the next node should be using smart routing with fallback.
        
        This method now uses the SmartRouter if available, otherwise falls back
        to the original rule-based routing logic.
        """
        # Try smart routing first if enabled and available
        if self.use_smart_routing and self.smart_router:
            try:
                next_node = self.smart_router.route_next(state)
                print(f"Smart router decision for conversation {state.conversation_id}: {next_node}")
                return next_node
            except Exception as e:
                log.warning(f"Smart router failed for conversation {state.conversation_id}: {e}. Using fallback.")
                # Continue to fallback routing below
        
        # Fallback to original rule-based routing
        log.info(f"Using rule-based routing for conversation {state.conversation_id}")
        return self._rule_based_routing(state)

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

        # 7.  Create message only if it's not a duplicate
        assistant_msg = create_message_if_not_duplicate(
            db_client=self.db_client,
            conversation_id=conversation_id,
            sender="assistant",
            kind=final_resp["kind"],
            final_resp=final_resp
        )

        return {
            "message": assistant_msg,
            "response": final_resp,
            "workflow_status": final.workflow_status.value,
            "steps_completed": len(final.steps_done),
        }

    # And update your process_user_file method:
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

        # 9. Create message only if it's not a duplicate
        assistant_msg = create_message_if_not_duplicate(
            db_client=self.db_client,
            conversation_id=conversation_id,
            sender="assistant",
            kind=final_resp["kind"],
            final_resp=final_resp
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