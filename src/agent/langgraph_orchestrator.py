# langraph_orchestrator.py
from typing import Dict, Any, List, Optional
import json

from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agent.agent_tools import get_tools
from src.agent.agent_state import AgentState, WorkflowStatus
from src.db.db_client import SupabaseClient

from src.db.checkpointer import load_agent_state, save_agent_state, _checkpoint_data_to_agent_state


class LangGraphOrchestrator:
    """
    LangGraph orchestrator for multi-step workflow management with manual Supabase persistence.
    """

    def __init__(
        self,
        llm,
        tools: Optional[List[BaseTool]] = None,
        db_client: Optional[SupabaseClient] = None
    ):
        self.llm = llm
        self.db_client = db_client

        if not db_client:
            raise ValueError("db_client is required for persistence")

        # Get or build tools
        if tools is None:
            self.tools = get_tools(db_client=db_client)
        else:
            self.tools = tools

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create ToolNode for LangGraph
        self.tool_node = ToolNode(self.tools)

        # Build (and compile) the workflow graph, WITHOUT attaching a checkpointer
        self.graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow. We do NOT attach a checkpointer here,
        because we'll handle persistence manually (load/save) around each run.
        """
        workflow = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("finalize", self._finalize_response)

        # ── Entry ──────────────────────────────────────────────
        workflow.set_entry_point("agent")

        # ── Edges ──────────────────────────────────────────────
        workflow.add_edge("tools", "finalize")
        workflow.add_edge("finalize", END)

        # Decide: either go to tools or finish
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": "finalize"},
        )

        # Compile WITHOUT any checkpointer
        return workflow.compile()

    def _create_system_message(self, state: AgentState) -> str:
        """
        Create a system prompt that reflects the current AgentState:
        which step we're on, which fields are filled, etc.
        """
        base_message = (
            "You are a specialized document translation assistant that helps users translate official documents "
            "(like PSA certificates, passports, IDs) into different languages using pre-built templates. "
            "Your goal is to guide users through a structured workflow that involves document analysis, "
            "field extraction, translation, and template filling.\n\n"
        )

        workflow_context = ""
        if state.user_upload_id and not state.template_id:
            workflow_context += (
                "The user has uploaded a document. Focus on analyzing the document type, variation, and source language. "
            )
        elif state.template_id and not state.filled_required_fields:
            workflow_context += (
                "A template has been identified. Focus on extracting required fields from the uploaded document. "
            )
        elif state.filled_required_fields and state.missing_required_fields:
            workflow_context += (
                "Some fields have been extracted but others are missing. Ask the user to provide the missing information. "
            )
        elif state.filled_required_fields and not state.missing_required_fields and not state.translated_required_fields:
            workflow_context += (
                "All required fields are filled. Focus on translating the fields to the target language. "
            )
        elif state.translated_required_fields and not state.document_version_id:
            workflow_context += (
                "Fields have been translated. Focus on filling the template and generating the final document. "
            )
        elif state.document_version_id:
            workflow_context += (
                "The translated document has been generated. Offer download links or further assistance. "
            )
        else:
            workflow_context += (
                "The user is starting a new translation workflow. Ask about their document type and target language, then show upload options. "
            )

        translation_info = ""
        if state.translate_from and state.translate_to:
            translation_info = f"Translation direction: {state.translate_from} → {state.translate_to}. "
        elif state.translate_to:
            translation_info = f"Target language: {state.translate_to}. "

        template_info = ""
        if state.template_id and state.template_required_fields:
            required_fields = list(state.template_required_fields.keys())
            truncated = ", ".join(required_fields[:5])
            elipsis = "..." if len(required_fields) > 5 else ""
            template_info = f"Using template requiring fields: {truncated}{elipsis}. "

        progress_info = ""
        if state.steps_done:
            progress_info = f"Completed steps: {', '.join(state.steps_done)}. "

        contextual_guidance = workflow_context + translation_info + template_info + progress_info

        instructions = (
            "WORKFLOW INSTRUCTIONS:\n"
            "- Always use tools for interactive elements (buttons, upload controls, forms)\n"
            "- Guide users step-by-step through: document upload → analysis → field extraction → translation → template filling\n"
            "- If fields are missing after extraction, create forms to collect the missing information\n"
            "- Always confirm extracted information with the user before proceeding\n"
            "- Use the steps_done list to avoid repeating completed workflow steps\n"
            "- Be clear about the current step and what comes next\n\n"
            "CURRENT CONTEXT:\n"
            + contextual_guidance
            + "\n"
            "RESPONSE GUIDELINES:\n"
            "- Reason step-by-step internally (do not reveal your internal thoughts)\n"
            "- Use tool calls when interaction is needed (buttons, forms, uploads)\n"
            "- Provide clear, professional updates on workflow status\n"
            "- Keep the user informed and move the workflow forward efficiently"
        )

        return base_message + instructions

    async def _agent_node(self, state: AgentState) -> AgentState:
        """
        Main agent node: build an LLM prompt (system + all prior messages),
        call the LLM (with tools), append its reply to state.messages.
        """
        try:
            if "agent" not in state.steps_done:
                state.steps_done.append("agent")

            llm_messages = [SystemMessage(content=self._create_system_message(state)), *state.messages]
            if not llm_messages:
                # If for some reason there are no prior messages, send a simple fallback
                default_response = AIMessage(content="I'm ready to help you with your request.")
                state.messages.append(default_response)
                state.workflow_status = WorkflowStatus.COMPLETED
                return state

            print("➤ Sending payload to LLM:")
            for idx, m in enumerate(llm_messages):
                txt = getattr(m, "content", "")
                print(f"  [{idx:2d}] ({type(m).__name__}): {repr(txt[:100])}  length={len(txt)}")

            response = await self.llm_with_tools.ainvoke(llm_messages)
            print("🧠  _agent_node – LLM replied:", response.content[:120] if hasattr(response, "content") else response)

            if getattr(response, "tool_calls", None):
                for call in response.tool_calls:
                    print("   ↳ tool requested:", call["name"], call["args"])

            state.messages.append(response)
            state.workflow_status = WorkflowStatus.IN_PROGRESS

        except Exception as e:
            print(f"[Error in _agent_node] {e}")
            error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
            state.messages.append(error_msg)
            state.workflow_status = WorkflowStatus.FAILED

        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Run any requested tool(s) and keep the full chat history."""
        try:
            if "tools" not in state.steps_done:
                state.steps_done.append("tools")

            prior_messages = list(state.messages)          # snapshot history

            if not state.messages:
                return state
            last = state.messages[-1]
            if not getattr(last, "tool_calls", None):
                return state

            for call in last.tool_calls:
                print(f"🔧 Calling {call['name']} with {call['args']}")

            # ---------- run the tool ----------
            raw_tool_resp = await self.tool_node.ainvoke(state)
            print("🔧  tool(s) finished")

            # ---------- normalise -------------
            if isinstance(raw_tool_resp, AgentState):
                merged_state = raw_tool_resp
            elif isinstance(raw_tool_resp, dict):
                merged_state = _checkpoint_data_to_agent_state(raw_tool_resp)
            else:                                   # AddableValuesDict etc.
                merged_state = _checkpoint_data_to_agent_state(dict(raw_tool_resp))

            # ---------- merge history ----------
            merged_state.messages = prior_messages + (merged_state.messages or [])
            return merged_state

        except Exception as e:
            print(f"[Error in _tool_node] {e}")
            err_msg = ToolMessage(
                content=f"Tool execution failed: {e}",
                tool_call_id="error"
            )
            state.messages = list(state.messages) + [err_msg]
            return state

        except Exception as e:
            print(f"[Error in _tool_node] {e}")
            # If tool execution fails, we still want to append a ToolMessage so the graph can move on.
            # First, restore full history:
            full_history = list(state.messages)

            # Create an error ToolMessage:
            last_msg = state.messages[-1] if state.messages else None
            err_tool_msg = ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id=(
                    "error"
                    if not last_msg or not getattr(last_msg, "tool_calls", None)
                    else last_msg.tool_calls[0].get("id", "error")
                )
            )

            # Append that single error msg onto our full history, then return new state:
            state.messages = full_history + [err_tool_msg]
            return state


    def _should_continue(self, state: AgentState) -> str:
        """
        Inspect the last message: if it has tool_calls, go to "tools"; otherwise "finalize".
        """
        if not state.messages:
            return "end"

        last_msg = state.messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f"🔧 Agent requested tools, routing to 'tools' node")
            return "tools"

        print(f"💬 Agent gave text response, routing to 'finalize' node")
        return "end"

    def _extract_final_response_from_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Look backwards through messages to find either:
         1) A ToolMessage whose content is JSON for UI elements (buttons, inputs, etc.)
         2) Or else the most recent AIMessage with plain text.
        Return a dict {"kind":"text","text": "..."} or the JSON content.
        """
        if not messages:
            return {"kind": "text", "text": "Task completed."}

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.content.strip():
                try:
                    obj = json.loads(msg.content)
                    if obj.get("kind") in ["buttons", "inputs", "file_card", "upload_button"]:
                        return obj
                except:
                    pass
                return {"kind": "text", "text": msg.content}

            elif isinstance(msg, AIMessage) and msg.content.strip():
                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    return {"kind": "text", "text": msg.content}

        return {"kind": "text", "text": "Task completed."}

    async def _finalize_response(self, state: AgentState) -> AgentState:
        """
        Final node: extract the “final_response” from state.messages (via _extract_final_response),
        store it in state.context["final_response"], then mark workflow as COMPLETED.
        """
        if "finalize" not in state.steps_done:
            state.steps_done.append("finalize")

        final_resp = self._extract_final_response_from_messages(state.messages)
        state.context["final_response"] = final_resp
        state.workflow_status = WorkflowStatus.COMPLETED

        print(f"Finalized: {final_resp}")
        return state

    async def process_user_message(
        self, conversation_id: str, user_message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        1) Load prior AgentState (if any) from Supabase.
        2) Append the new user_message as a HumanMessage.
        3) Run the LangGraph workflow (graph.ainvoke).
        4) If the returned object is a dict-like (AddableValuesDict), convert it into AgentState.
        5) Immediately save the updated AgentState back to Supabase.
        6) Return the “final_response” for the frontend to display.
        """
        # 1) Attempt to load existing AgentState
        prior_state = load_agent_state(self.db_client, conversation_id)

        if prior_state:
            print(f"[process_user_message] Loaded existing state for {conversation_id}")
            state = prior_state
            state.messages.append(HumanMessage(content=user_message))
            state.context.update(context)
        else:
            print(f"[process_user_message] No prior state. Creating new AgentState for {conversation_id}")
            state = AgentState(
                messages=[HumanMessage(content=user_message)],
                conversation_history=[],
                conversation_id=conversation_id,
                user_id=context.get("user_id", ""),
                workflow_status=WorkflowStatus.PENDING,
                context=context,
                user_upload_id="",
                user_upload={},
                extracted_required_fields={},
                filled_required_fields={},
                translated_required_fields={},
                missing_required_fields={},
                translate_from="",
                translate_to="",
                template_id="",
                template_required_fields={},
                document_version_id="",
                steps_done=[],
            )

        try:
            # 3) Run the LangGraph workflow (in-place modifies `state`)
            raw_result = await self.graph.ainvoke(state, RunnableConfig({}))

            # 4) If LangGraph returned a dict-like instead of AgentState, convert it:
            if isinstance(raw_result, AgentState):
                final_state = raw_result
            elif isinstance(raw_result, dict):
                final_state = _checkpoint_data_to_agent_state(raw_result)
            else:
                # e.g. AddableValuesDict—cast to plain dict then deserialize
                final_state = _checkpoint_data_to_agent_state(dict(raw_result))

            print("[process_user_message] Workflow completed. Saving new state…")
            # 5) Save the new state back to Supabase
            saved_ok = save_agent_state(self.db_client, conversation_id, final_state)
            if not saved_ok:
                print("[process_user_message] Warning: failed to save state")

            # 6) Extract final_response
            final_resp = final_state.context.get("final_response")
            if not final_resp:
                final_resp = self._extract_final_response_from_messages(final_state.messages)

            # Persist the assistant’s outgoing message into your “messages” table
            assistant_msg = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind=final_resp["kind"],
                body=json.dumps(final_resp),
            )

            return {
                "message": assistant_msg,
                "response": final_resp,
                "workflow_status": final_state.workflow_status.value,
                "steps_completed": len(final_state.steps_done),
            }

        except Exception as e:
            print(f"[process_user_message] Error in workflow: {e}")
            err_resp = {"kind": "text", "text": f"I encountered an error: {str(e)}"}
            assistant_msg = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps(err_resp),
            )
            return {
                "message": assistant_msg,
                "response": err_resp,
                "workflow_status": WorkflowStatus.FAILED.value,
                "steps_completed": 0,
            }

    async def handle_user_action(
        self, conversation_id: str, action: str, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert button clicks / file uploads / form submissions into a natural-text
        “I clicked X” or “I uploaded Y” message, then call process_user_message().
        """
        try:
            print(f"[handle_user_action] action={action}, values={values}")

            if action.startswith("button_") or action in ["yes", "no", "confirm", "cancel"]:
                if "label" in values:
                    button_text = values["label"]
                elif "value" in values:
                    button_text = values["value"]
                elif "text" in values:
                    button_text = values["text"]
                else:
                    button_text = action.replace("button_", "").replace("_", " ").title()

                msg = f"My answer is: {button_text}"
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "button_context": {
                        "selected_text": button_text,
                        "selected_action": action,
                        "is_answer": True,
                    },
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            elif action == "file_uploaded":
                file_info = values.get("file_info", {})
                f_name = file_info.get("name", "Unknown file")
                f_size = file_info.get("size", "Unknown size")
                f_type = file_info.get("type", "Unknown type")
                msg = f"I uploaded a file: {f_name} ({f_size}, {f_type})"
                print(f"[handle_user_action] File upload message: {msg}")
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "file_info": file_info,
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            elif action == "form_submitted":
                form_data = values.get("form_data", {})
                parts = [f"{k}: {v}" for k, v in form_data.items()]
                msg = "I submitted the form with: " + ", ".join(parts)
                print(f"[handle_user_action] Form submission message: {msg}")
                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "form_data": form_data,
                }
                return await self.process_user_message(conversation_id, msg, ctx)

            else:
                print(f"[handle_user_action] Unknown action. Fallback to generic text.")
                if "message" in values:
                    msg = values["message"]
                else:
                    msg = f"I selected: {action}"

                ctx = {
                    "user_id": values.get("user_id", ""),
                    "conversation_id": conversation_id,
                    "source_action": action,
                    "action_values": values,
                }
                return await self.process_user_message(conversation_id, msg, ctx)

        except Exception as e:
            print(f"[handle_user_action] Error: {e}")
            err_resp = {"kind": "text", "text": f"I encountered an error processing your action: {str(e)}"}
            assistant_msg = self.db_client.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                kind="text",
                body=json.dumps(err_resp),
            )
            return {
                "message": assistant_msg,
                "response": err_resp,
                "workflow_status": WorkflowStatus.FAILED.value,
                "steps_completed": 0,
            }

    def get_workflow_status(self, conversation_id: str) -> Optional[str]:
        """
        Return the current workflow_status for a conversation by loading the saved AgentState.
        """
        try:
            state = load_agent_state(self.db_client, conversation_id)
            if state:
                return state.workflow_status.value
            return None
        except Exception as e:
            print(f"[get_workflow_status] Error: {e}")
            return None

    def clear_conversation_state(self, conversation_id: str) -> bool:
        """
        Delete the saved AgentState row from Supabase, resetting the conversation.
        """
        try:
            self.db_client.client.table("agent_state").delete().eq("conversation_id", conversation_id).execute()
            return True
        except Exception as e:
            print(f"[clear_conversation_state] Error: {e}")
            return False
