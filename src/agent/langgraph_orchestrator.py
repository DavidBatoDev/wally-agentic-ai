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

from src.agent.agent_state import AgentState, WorkflowStatus, FieldMetadata
from agent.core_tools import get_tools
from src.db.checkpointer import (
    _checkpoint_data_to_agent_state,
    load_agent_state,
    save_agent_state,
)
from src.db.workflows_db import save_current_document_in_workflow_state
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
                print("Smart router initialized successfully")
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
        g.add_node("find_template",             self._find_template_node)
        g.add_node("extract_required_fields",   self._extract_values_node)
        g.add_node("ask_user_desired_langauge", self._ask_user_desired_language_node)

        # ── 2. Linear backbone you can already run ────────────────────────
        g.set_entry_point("load_state_graph")

        # After loading state, check if we need to handle a file upload
        g.add_conditional_edges(
            "load_state_graph",
            self._handle_upload,
            {
                "analyze_doc": "analyze_doc",  # File upload detected -> analyze first
                "agent": "agent",              # No file upload -> normal agent flow
            }
        )

        g.add_conditional_edges(
            "analyze_doc",
            self._is_document_templatable, # will implenent this if latest_user_upload is templatable by getting the latest_user_upload.is_templatable == True go to _ask_user_desired_language_node if only the state.translated_to is empty
            {
                "has_desired_user_desired_language": "find_template",  # File upload detected -> analyze first
                "dont_have_desired_language": "ask_user_desired_langauge",
                "agent": "agent", # if latest_user_upload.is_templatable == False     
            }
        )

        g.add_edge("ask_user_desired_langauge", "save_state_graph")

        # Generic tool-invocation → save
        g.add_edge("tools",            "save_state_graph")
        g.add_edge("find_template",            "agent")
        g.add_edge("extract_required_fields",  "save_state_graph")

        # Always persist then finalise
        g.add_edge("save_state_graph", "finalize")
        g.add_edge("finalize",         END)

        # ── 3. Dynamic router after *every* agent turn ─────────────────────
        g.add_conditional_edges(
            "agent",
            self._route_next, # your smart router
            {
                "find_template":           "find_template",
                "extract_required_fields":   "extract_required_fields", # if we want to get the values of extract_required_fields using the required_fields of the template
                "ask_user_desired_langauge": "ask_user_desired_langauge", # when we want to ask the user for the desired language
                "tools":                    "tools", # when we want to call any tools like: showing_buttons, showing_upload_file_button
                "save":                     "save_state_graph", # we want to end the current workflow but set the workflow-status to in progress
            },
        )

        return g.compile()

    # Conditional Edge
    def _is_document_templatable(self, state: AgentState) -> str:
        """
        Conditional routing logic after document analysis.
        
        Checks if the document is templatable and if user has specified desired language.
        Routes to:
        - "has_desired_user_desired_language": Document is templatable AND translate_to is set
        - "dont_have_desired_language": Document is templatable BUT translate_to is empty
        - "agent": Document is not templatable (fallback to normal agent flow)
        """
        print(f"🔍 [IS_DOC_TEMPLATABLE] Checking templatable status for conversation: {state.conversation_id}")
        
        # Check if we have a latest upload with analysis
        if not state.latest_upload or not state.latest_upload.analysis:
            print(f"🔍 [IS_DOC_TEMPLATABLE] No upload or analysis found, routing to agent")
            return "agent"
        
        # Get templatable status from analysis
        is_templatable = state.latest_upload.analysis.get("is_templatable", False)
        
        print(f"🔍 [IS_DOC_TEMPLATABLE] Document templatable: {is_templatable}")

        
        if not is_templatable:
            # Document is not templatable, route to normal agent flow
            print(f"🔍 [IS_DOC_TEMPLATABLE] Document not templatable, routing to agent")
            return "agent"
        
        print(f"🔍 [IS_DOC_TEMPLATABLE] Current translate_to: '{state.translate_to}'")
        
        # Check if user has specified desired translation language
        if state.translate_to and state.translate_to.strip():
            # User has specified desired language, proceed with template matching
            print(f"🔍 [IS_DOC_TEMPLATABLE] Has desired language ({state.translate_to}), routing to find_template")
            return "has_desired_user_desired_language"
        else:
            # User hasn't specified desired language, ask for it
            print(f"🔍 [IS_DOC_TEMPLATABLE] No desired language specified, routing to ask for language")
            return "dont_have_desired_language"

    def _handle_upload(self, state: AgentState) -> str:
        """
        Check if the latest message indicates a file upload.
        If so, route to analyze_doc_node first. Otherwise, go to agent.
        """
        if not state.messages:
            return "agent"
        
        latest_message = state.messages[-1]
        
        # Check if it's a HumanMessage with file upload indicator
        if (isinstance(latest_message, HumanMessage) and 
            latest_message.content.startswith("[FILE_UPLOADED]")):
            print(f"🔍 [HANDLE_UPLOAD] File upload detected for conversation: {state.conversation_id}")
            return "analyze_doc"
        
        # Default to agent node for regular messages
        return "agent"

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

    # System Prompt
    def _create_system_message(self, state: AgentState) -> str:
        """Return a dynamic system prompt describing the current workflow step."""
        base = (
            "You are a specialised document-translation assistant that guides the "
            "user through a structured flow: upload ➜ analyze ➜ ask language ➜ find template "
            "➜ extract fields ➜ translate ➜ generate final document.\n\n"
        )

        ctx: List[str] = []
        current_doc = state.current_document_in_workflow_state
        
        # Determine current workflow stage based on updated state structure
        if state.latest_upload and state.latest_upload.is_templatable and not state.translate_to:
            ctx.append("Document uploaded and analyzed as templatable. Ask user for desired translation language.")
        elif state.translate_to and not current_doc.template_id:
            ctx.append("Target language specified. Find matching template based on document analysis.")
        elif current_doc.template_id and not current_doc.fields:
            ctx.append("Template found. Extract required field values from the uploaded document.")
        elif current_doc.fields and state.translate_to:
            # Check if any fields need translation
            needs_translation = any(
                field.translated_status == "pending" 
                for field in current_doc.fields.values()
            )
            if needs_translation:
                ctx.append("Fields extracted. Translate field values to the target language.")
            elif not current_doc.current_document_version_public_url:
                ctx.append("Fields translated. Generate the final translated document.")
        elif current_doc.current_document_version_public_url:
            ctx.append("Final translated document generated. Workflow complete - offer download or further assistance.")
        else:
            if not state.latest_upload:
                ctx.append("Ready to start new document translation workflow.")
            else:
                ctx.append("Introduce yourself and explain the current step in the workflow.")

        # Add contextual information
        if state.translate_to:
            ctx.append(f"Target language: {state.translate_to}.")
        
        if state.latest_upload and state.latest_upload.analysis and state.latest_upload.is_templatable:
            analysis = state.latest_upload.analysis
            doc_type = analysis.get('doc_type', 'unknown')
            variation = analysis.get('variation', 'unknown')
            detected_lang = analysis.get('detected_language', 'unknown')
            ctx.append(f"Document analyzed: {doc_type} ({variation}) in {detected_lang}.")
        
        if current_doc.template_required_fields:
            keys = list(current_doc.template_required_fields.keys())[:5]
            more = "..." if len(current_doc.template_required_fields) > 5 else ""
            ctx.append(f"Template requires: {', '.join(keys)}{more}.")
        
        if current_doc.fields:
            filled_keys = [k for k, v in current_doc.fields.items() if v.value_status != "pending"][:3]
            if filled_keys:
                more = "..." if len(filled_keys) > 3 else ""
                ctx.append(f"Fields with values: {', '.join(filled_keys)}{more}.")

        # Add workflow status context
        if state.workflow_status != WorkflowStatus.PENDING:
            ctx.append(f"Workflow status: {state.workflow_status.value}.")
        
        if state.current_pending_node:
            ctx.append(f"Pending action: {state.current_pending_node}.")

        guidelines = (
            "GUIDELINES:\n"
            "- Use tool calls for UI elements (buttons, forms, uploads, file displays).\n"
            "- You can call MULTIPLE tools in a single response when needed.\n"
            "- For example: call update_agent_state AND show_buttons together.\n"
            "- Never repeat steps already in steps_done.\n"
            "- The smart router will determine the next workflow node based on current state and conversation.\n"
            "- Focus on clear communication about the current step and what the user needs to do.\n"
            "- Handle user corrections and field modifications gracefully.\n"
            "- Provide professional updates about workflow progress.\n"
            f"- ALWAYS include conversation_id='{state.conversation_id}' when calling tools that support it.\n"
        )

        # Updated tool hint reflecting new workflow nodes
        tool_hint = (
            "\n\nIf the user provides ANY new workflow information "
            "(translate_to, field corrections, language preferences), **call the tool "
            "`update_agent_state`** with only the keys that changed. "
            "The main updatable fields are: translate_to, workflow_status, current_pending_node, and field values. "
            "You can then ALSO call other tools like show_buttons, show_upload_button, "
            "or show_document_preview in the same response to provide the next UI interaction. "
            f"Remember to always pass conversation_id='{state.conversation_id}' to tools. "
            "After all tool calls, send your normal assistant reply explaining the current step."
        )

        return (
            base
            + f"Current state: [{self._state_summary(state)}]. "
            + " ".join(ctx)
            + "\n\n"
            + guidelines
            + tool_hint
        )

    # Nodes
    async def _agent_node(self, state: AgentState) -> AgentState:
        # Mark that _agent_node has run
        if "agent" not in state.steps_done:
            state.steps_done.append("agent")

        # 1) Collect only non-tool, non-empty messages
        clean_messages = []
        for msg in state.messages:
            # If it's an AIMessage and has tool_calls present
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                # Case A: Both content and tool_calls are non‐empty
                if msg.content and msg.content.strip():
                    
                    # 1) Tool‐only message (tool_calls must be a list)
                    tool_msg = AIMessage(
                        content="",  # explicitly empty string
                        tool_calls=list(msg.tool_calls),  # ensure this is a list
                        additional_kwargs=msg.additional_kwargs,
                        response_metadata=msg.response_metadata
                    )
                    clean_messages.append(tool_msg)

                else:
                    continue
                    # Case B: No content, only tool_calls (still wrap in list)
                    tool_msg = AIMessage(
                        content="",
                        tool_calls=list(msg.tool_calls),  # wrap in list
                        additional_kwargs=msg.additional_kwargs,
                        response_metadata=msg.response_metadata
                    )
                    clean_messages.append(tool_msg)

                continue
            elif isinstance(msg, AIMessage) and msg.content.strip() == "":
                # If it's an AIMessage with empty content, skip it
                continue

            # Everything else (e.g., HumanMessage or AIMessage without tool_calls)
            clean_messages.append(msg)

        for msg in clean_messages:
            if msg.content.strip() == "":
                print(f"🙏🏻 MSG content: ", msg)

        # 2) Build the SystemMessage with the current workflow context
        sys_msg = SystemMessage(content=self._create_system_message(state))

        # 3) Prepend the system prompt to the clean chat history
        llm_messages: List[BaseMessage] = [sys_msg, *clean_messages]

        # 4) Invoke the LLM (which may emit one or more tool_calls)
        response: AIMessage = await self.llm_with_core_tools.ainvoke(llm_messages)

        # 5) If the LLM produced any "update_agent_state" tool_calls, apply them
        for tc in getattr(response, "tool_calls", []):
            if tc["name"] == "update_translate_to":
                patch = {k: v for k, v in tc["args"].items() if k != "conversation_id"}
                self._apply_state_patch(state, patch)

        # 6) Append the LLM's response to the state

        print("🤖🙏🏻 MSG response", response)
        
        # Create message in database
        if response.content.strip() != "":
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({"text": response.content}),
            )


        state.messages.append(response)
        state.workflow_status = WorkflowStatus.IN_PROGRESS
        return state

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
        analysis_msg = AIMessage(content="🔍 Analyzing the uploaded file")
        state.messages.append(analysis_msg)
        
        # Create message in database too
        self.db_client.create_message(
            conversation_id=state.conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps({
                "text": "🔍 Analyzing...."
            }),
        )
        # Mark this step as done
        state.steps_done.append("analyze_doc")

        try:
            print(f"🔍 [ANALYZE_DOC] Calling analyze tool directly...")

            # import analyze_tool result
            from agent.functions.analyze_doc_node_helper import analyze_upload

            # Check if we have a valid upload with public_url
            if not state.latest_upload:
                raise ValueError("No normal upload found for document analysis")
            
            if not state.latest_upload.public_url:
                raise ValueError("Missing required upload URL for document analysis")
            
            if state.latest_upload.public_url == "" or state.latest_upload.public_url is None:
                raise ValueError("Missing required upload URL for document analysis")

            analysis_result = await analyze_upload(
                file_id=state.latest_upload.file_id,
                public_url=state.latest_upload.public_url
            )
            
            # Store the analysis in state
            state.latest_upload.analysis = {**analysis_result}
            
            print(f"🔍 [ANALYZE_DOC] Stored analysis in state")
            
            # Check if document is templatable and move to appropriate list
            is_templatable = analysis_result.get("is_templatable", False)
            state.latest_upload.is_templatable = is_templatable
            
            # Create response based on templatable status
            if is_templatable:
                # Create detailed summary for templatable documents
                doc_type = analysis_result.get("doc_type", "unknown")
                variation = analysis_result.get("variation", "standard")
                language = analysis_result.get("detected_language", "unknown")
                doc_classification = analysis_result.get("doc_classification", "other")

                summary_parts = [
                    f"✅ **Document Analysis Complete**",
                    f"• **Type**: {doc_type.replace('_', ' ').title()}\n",
                    f"• **Classification**: {doc_classification.replace('_', ' ').title()}\n",
                    f"• **Variation**: {variation.replace('_', ' ').title()}\n",
                    f"• **Language**: {language.replace('_', ' ').title()}\n",
                    f"• **Templatable**: {'Yes' if is_templatable else 'No'}\n",
                ]
                
                if analysis_result.get("page_count"):
                    summary_parts.append(f"• **Pages**: {analysis_result['page_count']}\n")
                if analysis_result.get("page_size"):
                    summary_parts.append(f"• **Size**: {analysis_result['page_size']}\n")

                upload_summary = analysis_result.get("content_summary", "")
                summary_parts.append(f"• **Summary**: {upload_summary}\n")
                
                response_text = "\n".join(summary_parts)
            else:
                # Simple response for non-templatable documents
                upload_summary = analysis_result.get("content_summary", "Document analyzed")
                response_text = f"{upload_summary}\n\nThis document appears to not be related to document translation or template processing. So I won't proceed to the document translation"
            
            summary_msg = AIMessage(content=response_text)
            state.messages.append(summary_msg)
            
            # Create message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": response_text
                }),
            )

            state.workflow_status = WorkflowStatus.IN_PROGRESS
            
            print(f"🔍 [ANALYZE_DOC] Analysis completed successfully")
            
        except Exception as e:
            print(f"❌ [ANALYZE_DOC] Error during analysis: {str(e)}")
            error_msg = AIMessage(content=f"❌ Error analyzing document: {str(e)}")
            state.messages.append(error_msg)
            
            # Create error message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": f"❌ Error analyzing document: {str(e)}"
                }),
            )
            
            state.workflow_status = WorkflowStatus.FAILED
        
        return state

    async def _find_template_node(self, state: AgentState) -> AgentState:
        """
        Use the find_template_match tool to find the best matching template.
        This calls the template matching tool directly in a deterministic way.
        """
        print(f"🔍 [FIND_TEMPLATE] Starting for conversation: {state.conversation_id}")

        # Insert a message that we're finding a template
        template_msg = AIMessage(content="🔍 Finding the best matching template for your document...")
        state.messages.append(template_msg)

        # Create message in database too
        self.db_client.create_message(
            conversation_id=state.conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps({
                "text": "🔍 Finding the best matching template for your document..."
            }),
        )

        # Mark this step as done
        state.steps_done.append("find_template")
        
        try:
            # Hardcoded template ID
            template_id = "9fc0c5fc-2885-4d58-ba0f-4711244eb7df"
            
            # Get template data from Supabase
            print(f"🔍 [FIND_TEMPLATE] Fetching template data for ID: {template_id}")
            
            template_response = self.db_client.client.table("templates").select(
                "id, doc_type, variation, file_url, info_json"
            ).eq("id", template_id).single().execute()
            
            if not template_response.data:
                raise Exception(f"Template with ID {template_id} not found")
            
            template_data = template_response.data
            
            # Extract required fields from info_json
            info_json = template_data.get("info_json", {})
            template_required_fields = info_json.get("required_fields", {})
            
            print(f"🔍 [FIND_TEMPLATE] Template found - Type: {template_data['doc_type']}, Variation: {template_data['variation']}")
            print(f"🔍 [FIND_TEMPLATE] Required fields: {list(template_required_fields.keys())}")
            
            # Find the latest user upload where templatable is true
            latest_templatable_upload = None
            for upload in reversed(state.user_uploads):  # Check from most recent
                if upload.is_templatable:
                    latest_templatable_upload = upload
                    break
            
            if not latest_templatable_upload:
                raise Exception("No templatable upload found in user uploads")
            
            print(f"🔍 [FIND_TEMPLATE] Using upload: {latest_templatable_upload.filename}")
            
            # Initialize fields with FieldMetadata for each required field
            fields = {}
            for field_key in template_required_fields.keys():
                fields[field_key] = FieldMetadata(
                    value="",                    # Empty value initially
                    value_status="pending",      # Status is pending
                    translated_value=None,       # No translation initially
                    translated_status="pending"  # Translation status is pending
                )
            
            # Update CurrentDocumentInWorkflow state
            state.current_document_in_workflow_state.file_id = latest_templatable_upload.file_id
            state.current_document_in_workflow_state.base_file_public_url = latest_templatable_upload.public_url
            state.current_document_in_workflow_state.template_id = template_id
            state.current_document_in_workflow_state.template_file_public_url = template_data["file_url"]
            state.current_document_in_workflow_state.template_required_fields = template_required_fields
            state.current_document_in_workflow_state.fields = fields  # Use the new fields structure
            
            # Create success message
            success_message = (
                f"✅ Found matching template!\n\n"
                f"📋 Document Type: {template_data['doc_type']}\n\n"
                f"🔧 Variation: {template_data['variation']}\n\n"
                f"📝 Required Fields: {len(template_required_fields)} fields to extract\n\n"
                f"📄 Processing: {latest_templatable_upload.filename}"
            )
            
            success_msg = AIMessage(content=success_message)
            state.messages.append(success_msg)
            
            # Create success message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": success_message
                }),
            )

            save_agent_state(self.db_client, state.conversation_id, state)
            save_current_document_in_workflow_state(self.db_client, state.conversation_id, state)
            
            print(f"🔍 [FIND_TEMPLATE] Successfully configured document workflow")
            print(f"🔍 [FIND_TEMPLATE] Initialized {len(fields)} fields with FieldMetadata structure")
            
        except Exception as e:
            print(f"🔍 [FIND_TEMPLATE] ERROR: {str(e)}")
            log.exception(f"Error finding template match: {e}")
        
            # Store empty template info on error
            state.current_document_in_workflow_state.template_id = ""
            state.current_document_in_workflow_state.template_required_fields = {}
            state.current_document_in_workflow_state.fields = {}  # Empty fields dict
        
            error_msg = AIMessage(content="⚠️ Unable to find a matching template. Manual processing may be required.")
            state.messages.append(error_msg)
        
            # Create error message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": "⚠️ Unable to find a matching template. Manual processing may be required."
                }),
            )
        
            state.workflow_status = WorkflowStatus.FAILED
            
        print(f"🔍 [FIND_TEMPLATE] Completed")
        return state

    async def _extract_values_node(self, state: AgentState) -> AgentState:
        """
        Extract values from the document using OCR based on template requirements.
        This calls the extract_values_from_document helper function directly.
        """
        print(f"📄 [EXTRACT_VALUES] Starting for conversation: {state.conversation_id}")
        
        # Insert a message that we're extracting values from the document
        extraction_msg = AIMessage(content="📄 Extracting values from the document using OCR...")
        state.messages.append(extraction_msg)
        
        # Create message in database too
        self.db_client.create_message(
            conversation_id=state.conversation_id,
            sender="assistant",
            kind="text",
            body=json.dumps({
                "text": "📄 Extracting values from the document using OCR..."
            }),
        )
        
        # Mark this step as done
        state.steps_done.append("extract_values")

        try:
            print(f"📄 [EXTRACT_VALUES] Calling extract values helper directly...")

            # Import the extract values helper
            from agent.functions.extract_values_node_helper import extract_values_from_document

            # Validate that we have the required data in state
            if not state.current_document_in_workflow_state.template_id:
                raise ValueError("No template ID found in current document workflow state")
            
            if not state.current_document_in_workflow_state.base_file_public_url:
                raise ValueError("No base file public URL found in current document workflow state")
            
            template_id = state.current_document_in_workflow_state.template_id
            base_file_url = state.current_document_in_workflow_state.base_file_public_url
            
            print(f"📄 [EXTRACT_VALUES] Using template_id: {template_id}")
            print(f"📄 [EXTRACT_VALUES] Using base_file_url: {base_file_url}")

            # Call the extraction function
            extraction_result = await extract_values_from_document(
                template_id=template_id,
                base_file_public_url=base_file_url
            )
            
            # Store the extracted fields in state using the new FieldMetadata structure
            if extraction_result.get("success", False):
                extracted_fields = extraction_result.get("extracted_ocr", {})
                
                # Update existing fields with extracted values
                for field_key, field_data in extracted_fields.items():
                    if field_key in state.current_document_in_workflow_state.fields:
                        # Update the existing FieldMetadata with the extracted value
                        state.current_document_in_workflow_state.fields[field_key].value = field_data.get("value", "")
                        state.current_document_in_workflow_state.fields[field_key].value_status = "ocr"
                    else:
                        # Create new FieldMetadata for fields not previously in the template
                        from agent.agent_state import FieldMetadata
                        state.current_document_in_workflow_state.fields[field_key] = FieldMetadata(
                            value=field_data.get("value", ""),
                            value_status="ocr"
                        )
                
                print(f"📄 [EXTRACT_VALUES] Successfully extracted {len(extracted_fields)} fields")
                print(f"📄 [EXTRACT_VALUES] Updated fields with OCR values: {len(extracted_fields)} fields")
                
                # Create detailed summary of extraction results
                missing_fields = extraction_result.get("missing_value_keys", {})
                
                doc_type = extraction_result.get("doc_type", "unknown")
                variation = extraction_result.get("variation", "standard")
                
                summary_parts = [
                    f"✅ **Value Extraction Complete**\n",
                    f"• **Document Type**: {doc_type.replace('_', ' ').title()}\n",
                    f"• **Variation**: {variation.replace('_', ' ').title()}\n\n",
                    f"• **Fields Extracted**: {len(extracted_fields)} out of {len(extracted_fields) + len(missing_fields)} total fields\n\n",
                ]
                
                # Show extracted fields summary
                if extracted_fields:
                    summary_parts.append("\n\n**📋 Successfully Extracted Fields:**\n\n")
                    for field_key, field_data in extracted_fields.items():
                        # Clean up the field key for display
                        clean_key = field_key.strip('{}')
                        label = field_data.get("label", clean_key)
                        # Get the value from the field_data dictionary
                        field_value = field_data.get("value", "")
                        # Truncate long values for display
                        display_value = str(field_value)[:50] + "..." if len(str(field_value)) > 50 else str(field_value)
                        summary_parts.append(f"• **{label}**: {display_value}\n\n")
                
                # Show missing fields if any
                if missing_fields:
                    summary_parts.append(f"\n\n**⚠️ Missing Fields ({len(missing_fields)}):**\n\n")
                    for field_key, field_data in missing_fields.items():
                        clean_key = field_key.strip('{}')
                        field_label = field_data.get("label", clean_key)
                        summary_parts.append(f"• **{clean_key}**: {field_label}\n\n")
                    summary_parts.append("\n\n*These fields were not found in the document or were unclear during OCR processing. Please manually add them*\n\n")
                
                # Join all parts and remove any trailing newlines, then ensure single trailing newline
                response_text = "".join(summary_parts).rstrip() + "\n\n"

                # Save the agent state with the extracted fields
                save_agent_state(self.db_client, state.conversation_id, state)
                
                # Set workflow status to waiting for confirmation
                state.workflow_status = WorkflowStatus.IN_PROGRESS
                
            else:
                # Handle extraction failure
                error_msg = extraction_result.get("error", "Unknown extraction error")
                response_text = f"❌ **Value Extraction Failed**\n\n\nError: {error_msg}"
                
                # If there's a raw response, include it for debugging
                if extraction_result.get("raw_response"):
                    response_text += f"\n\n*Debug Info*: {extraction_result['raw_response'][:200]}..."
                
                # Clear all field values on failure but keep the field structure
                for field_key in state.current_document_in_workflow_state.fields:
                    state.current_document_in_workflow_state.fields[field_key].value = ""
                    state.current_document_in_workflow_state.fields[field_key].value_status = "pending"
                
                state.workflow_status = WorkflowStatus.FAILED
            
            # Create the response message
            summary_msg = AIMessage(content=response_text)
            state.messages.append(summary_msg)
            
            # Create message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": response_text
                }),
            )

            caution_msg_text = "Please double check the values extracted from the document using our interactive document forms editor at the upper right. "
            caution_ai_msg = AIMessage(content=caution_msg_text)
            state.messages.append(caution_ai_msg)

            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": caution_msg_text
                }),
            )
            
            print(f"📄 [EXTRACT_VALUES] Extraction process completed")
            
        except Exception as e:
            print(f"❌ [EXTRACT_VALUES] Error during extraction: {str(e)}")
            error_msg = AIMessage(content=f"❌ Error extracting values from document: {str(e)}")
            state.messages.append(error_msg)
            
            # Create error message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": f"❌ Error extracting values from document: {str(e)}"
                }),
            )
            
            # Clear all field values on error but keep the field structure
            for field_key in state.current_document_in_workflow_state.fields:
                state.current_document_in_workflow_state.fields[field_key].value = ""
                state.current_document_in_workflow_state.fields[field_key].value_status = "pending"
            
            state.workflow_status = WorkflowStatus.FAILED
        
        return state

    async def _finalize_response(self, state: AgentState) -> AgentState:
        if "finalize" not in state.steps_done:
            state.steps_done.append("finalize")

        state.context["final_response"] = self._extract_final_response(state.messages)
        if state.workflow_status != WorkflowStatus.FAILED:
            state.workflow_status = WorkflowStatus.IN_PROGRESS
        return state

    async def _ask_user_desired_language_node(self, state: AgentState) -> AgentState:
        """
        Ask the user what language they want to translate the document to.
        This node is reached when we have a templatable document but no translate_to language specified.
        """
        print(f"🌐 [ASK_LANGUAGE] Starting for conversation: {state.conversation_id}")
        
        # Mark this step as done
        if "ask_user_desired_langauge" not in state.steps_done:
            state.steps_done.append("ask_user_desired_langauge")
        
        try:
            # Get document info for context
            doc_type = "document"
            detected_language = "unknown"
            
            if state.latest_upload and state.latest_upload.analysis:
                analysis = state.latest_upload.analysis
                doc_type = analysis.get("doc_type", "document").replace("_", " ").title()
                detected_language = analysis.get("detected_language", "unknown").replace("_", " ").title()
            
            # Create message asking for desired language
            language_prompt = (
                f"🌐 **Let me try to find existing template from the database**\n\n"
                f"I've analyzed your **{doc_type}** (currently in **{detected_language}**). Before I find a template document to the database please provide me a to target desired language!\n\n"
                f"**To try and find the template of this document, I would need to know the desired language, What language would you like me to translate it to?**\n\n"
                f"Please specify your desired target language, and I'll proceed with finding the appropriate template."
            )
            
            language_msg = AIMessage(content=language_prompt)
            state.messages.append(language_msg)
            
            # Create message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": language_prompt
                }),
            )
            
            # Set workflow status to waiting for user input
            state.workflow_status = WorkflowStatus.WAITING_CONFIRMATION
            
            print(f"🌐 [ASK_LANGUAGE] Language prompt sent successfully")
            
        except Exception as e:
            print(f"🌐 [ASK_LANGUAGE] ERROR: {str(e)}")
            log.exception(f"Error asking for desired language: {e}")
            
            error_msg = AIMessage(content="⚠️ Please specify what language you'd like me to translate your document to.")
            state.messages.append(error_msg)
            
            # Create error message in database
            self.db_client.create_message(
                conversation_id=state.conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps({
                    "text": "⚠️ Please specify what language you'd like me to translate your document to."
                }),
            )
            
            state.workflow_status = WorkflowStatus.WAITING_CONFIRMATION

        print(f"🌐 [ASK_LANGUAGE] Completed")
        return state
    
    # Helper Node
    def _merge_states(self, current_state: AgentState, db_state: AgentState) -> AgentState:
        """
        Merge the in-flight state (current_state) with the persisted copy (db_state).

        Rules
        -----
        • New data in current_state always wins.  
        • Lists → append only the items that are truly new.  
        • Dicts → shallow-update (current keys overwrite, db keys preserved).  
        • Nested `CurrentDocumentInWorkflow` → merge field-by-field.  
        """
        merged = db_state.model_copy(deep=True)

        # ── 1.  Messages & conversation history ────────────────────────────
        merged.messages.extend(current_state.messages[len(db_state.messages):])
        merged.conversation_history.extend(
            current_state.conversation_history[len(db_state.conversation_history):]
        )

        # ── 2.  Context ─────────────────────────────────────────────────────
        merged.context.update(current_state.context)

        # ── 3.  Uploads ─────────────────────────────────────────────────────
        if current_state.user_uploads:
            merged.user_uploads.extend(
                current_state.user_uploads[len(db_state.user_uploads):]
            )

        # ── 4.  Top-level scalars ───────────────────────────────────────────
        if current_state.translate_to:
            merged.translate_to = current_state.translate_to

        if current_state.current_pending_node:
            merged.current_pending_node = current_state.current_pending_node

        # ── 5.  Steps done ──────────────────────────────────────────────────
        for step in current_state.steps_done:
            if step not in merged.steps_done:
                merged.steps_done.append(step)

        # ── 6.  Workflow status (let failures / completions bubble up) ──────
        if current_state.workflow_status != merged.workflow_status:
            merged.workflow_status = current_state.workflow_status

        # ── 7.  CurrentDocumentInWorkflow merge ────────────────────────────
        cur_doc   = current_state.current_document_in_workflow_state
        merged_doc = merged.current_document_in_workflow_state

        # simple scalars
        for attr in (
            "file_id",
            "base_file_public_url",
            "template_id",
            "template_file_public_url",
            "current_document_version_public_url",
        ):
            val = getattr(cur_doc, attr, "")
            if val and val != getattr(merged_doc, attr):
                setattr(merged_doc, attr, val)

        # translate_to inside the nested doc (separate from the top-level one)
        if cur_doc.translate_to:
            merged_doc.translate_to = cur_doc.translate_to

        # dicts
        merged_doc.template_required_fields.update(cur_doc.template_required_fields)

        # FieldMetadata dict
        if cur_doc.fields:
            for key, field_md in cur_doc.fields.items():
                if key not in merged_doc.fields:
                    # brand-new field
                    merged_doc.fields[key] = field_md
                else:
                    # update existing field – keep the latest non-pending info
                    existing = merged_doc.fields[key]
                    if field_md.value_status != "pending":
                        existing.value = field_md.value
                        existing.value_status = field_md.value_status
                    if field_md.translated_status != "pending":
                        existing.translated_value = field_md.translated_value
                        existing.translated_status = field_md.translated_status

        return merged

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

    def _state_summary(self, state: AgentState) -> str:
        """Return a one-liner describing the high-level workflow state."""
        current_doc = state.current_document_in_workflow_state
        parts: list[str] = [f"Status: {state.workflow_status.value}"]

        # ── Upload info ────────────────────────────────────────────────────
        if state.latest_upload:
            parts.append("Has upload")
            if state.latest_upload.is_templatable:
                parts.append("templatable")

        # ── Desired language ───────────────────────────────────────────────
        if state.translate_to:
            parts.append(f"→ {state.translate_to}")

        # ── Template ───────────────────────────────────────────────────────
        if current_doc.template_id:
            parts.append("template found")

        # ── Field progress (new FieldMetadata-based logic) ─────────────────
        if current_doc.fields:
            ocr_cnt        = sum(1 for f in current_doc.fields.values() if f.value_status == "ocr")
            filled_cnt     = sum(1 for f in current_doc.fields.values() if f.value_status in {"edited", "confirmed"})
            translated_cnt = sum(1 for f in current_doc.fields.values() if f.translated_status in {"translated", "edited", "confirmed"})

            if ocr_cnt:
                parts.append(f"{ocr_cnt} fields OCR")
            if filled_cnt:
                parts.append(f"{filled_cnt} fields filled")
            if translated_cnt:
                parts.append(f"{translated_cnt} fields translated")

        # ── Final document ─────────────────────────────────────────────────
        if current_doc.current_document_version_public_url:
            parts.append("document ready")

        return " | ".join(parts)

    def _apply_state_patch(self, state: AgentState, patch: Dict[str, Any]) -> None:
        """
        Apply the partial update coming from `update_agent_state` tool calls.
        Simplified to handle only the explicit fields we want to support.
        """
        for key, val in patch.items():
            if not hasattr(state, key):
                # Silently skip anything not on AgentState
                continue

            # Default behaviour: let Pydantic handle the assignment
            setattr(state, key, val)

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
        state.user_uploads.append(file_info)

        # 4. Merge any additional context
        if context:
            state.context.update(context)
        
        
        # Also add file info to context for tools
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
