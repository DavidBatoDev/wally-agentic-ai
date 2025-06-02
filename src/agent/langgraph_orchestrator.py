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
from src.db.db_client import SupabaseClient
import re
from urllib.parse import unquote


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

        # Bind core_tools to the LLM and wrap them in a ToolNode runner
        self.llm_with_core_tools = self.llm.bind_tools(self.core_tools)
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
            "- You can call MULTIPLE tools in a single response when needed.\n"
            "- For example: call update_agent_state AND show_buttons together.\n"
            "- Never repeat steps already in steps_done.\n"
            "- Provide clear, professional updates.\n"
            "- Reason internally; do not expose chain-of-thought.\n"
            f"- ALWAYS include conversation_id='{state.conversation_id}' when calling tools that support it.\n"
        )

        tool_hint = (
            "\n\nIf the user message gives you ANY new detail about the workflow "
            "(language, template_id, document type, etc.) **call the tool "
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
        VERY simple stub:
        • infers doc_type / variation / language from the uploaded file name & mime-type
        • writes the result into state.upload_analysis
        • tells the user what we found
        """
        if "analyze_doc" in state.steps_done:
            # Already done in a previous turn
            return state

        state.steps_done.append("analyze_doc")

        analysis = {
            "doc_type":         " guess_doc_type(stem)",
            "variation":         "PSA",
            "detected_language": "English",
            "template_id_hint":  None,
        }

        # ------------------------------------------------------------------
        # 3) Patch the state
        state.upload_analysis = analysis

        # ------------------------------------------------------------------
        # 4) Let the user know
        bullet = [
            f"✅ File **PSA** received.",
            f"• Detected document type: **{analysis['doc_type']}**",
        ]
        if analysis["variation"]:
            bullet.append(f"• Variation / year: **{analysis['variation']}**")
        if analysis["detected_language"] != "unknown":
            bullet.append(f"• Language: **{analysis['detected_language']}**")

        state.messages.append(
            AIMessage(content="\n".join(bullet))
        )

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
        
        # Add file info to context so tools can access it
        state.context.update({
            "uploaded_file": file_info,
            "file_id": file_info.get("file_id"),
            "public_url": file_info.get("public_url"),
        })

        # 5. Run the state-graph
        raw = await self.graph.ainvoke(state, RunnableConfig())
        final: AgentState = (
            raw
            if isinstance(raw, AgentState)
            else _checkpoint_data_to_agent_state(dict(raw))
        )

        # 6. Persist the final state to Supabase
        save_agent_state(self.db_client, conversation_id, final)

        # 7. Figure out what "final_response" to send to the UI
        final_resp = final.context.get("final_response") or self._extract_final_response(
            final.messages
        )

        # 8. Insert that final payload into the messages table
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

    async def handle_user_action(
        self, conversation_id: str, action: str, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert button clicks / uploads / form submissions into natural-language
        messages so the main orchestrator can process them.
        """
        log.debug("handle_user_action action=%s values=%s", action, values)

        try:
            if action.startswith("button_") or action in {"yes", "no", "confirm", "cancel"}:
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
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                }
                return await self.process_user_file(conversation_id, info, ctx)

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

        • Ignore unknown keys so a mis-typed field never breaks the workflow.
        • Accept `user_upload_info` as dict or free-text; coerce to dict when possible.
        """
        for key, val in patch.items():
            if not hasattr(state, key):
                # Silently skip anything not on AgentState
                continue

            # ─────────────────────────────── user_upload_info
            if key == "user_upload_info":
                info_dict: Dict[str, Any] = {}

                if isinstance(val, dict):
                    info_dict = val

                elif isinstance(val, str):
                    # Try to parse "k=v, k=v, …" patterns first
                    for piece in val.split(","):
                        if "=" not in piece:
                            continue
                        k, v = (p.strip() for p in piece.split("=", 1))
                        k = k.lower()

                        v = re.sub(r"^['\"]?file[_ ]?type['\"]?:\s*", "", v).strip(" '\"")

                        match k:
                            case "filename" | "file_name":
                                info_dict["filename"] = v
                            case "filetype" | "mime_type" | "mimetype":
                                info_dict["mime_type"] = v
                            case "filesize" | "size_bytes" | "size":
                                num = re.sub(r"[^\d]", "", v)
                                if num.isdigit():
                                    info_dict["size_bytes"] = int(num)
                            case "url" | "public_url":
                                info_dict["public_url"] = unquote(v)
                            case "file_id":
                                info_dict["file_id"] = v

                    # Second chance: handle things like "application/pdf, 2 810 447 bytes"
                    if not info_dict:
                        tokens = [t.strip() for t in val.split(",")]
                        for t in tokens:
                            if "/" in t and "mime_type" not in info_dict:
                                info_dict["mime_type"] = t
                            elif "byte" in t.lower():
                                num = re.sub(r"[^\d]", "", t)
                                if num.isdigit():
                                    info_dict["size_bytes"] = int(num)

                    if not info_dict:
                        log.warning("Could not parse user_upload_info string: %s", val)
                        continue  # give up – don’t poison state

                else:
                    log.warning("Ignored invalid user_upload_info patch (%s)", type(val))
                    continue

                # Persist the cleaned info & sync convenience mirror
                state.user_upload_info = info_dict
                state.user_upload_id = info_dict.get("file_id", state.user_upload_id)
                continue  # done with this key

            # ─────────────────────────────── simple type guard
            if key == "user_upload_id" and not isinstance(val, str):
                log.warning("user_upload_id must be str – got %s", type(val))
                continue

            # Default behaviour: let Pydantic handle the assignment
            setattr(state, key, val)
