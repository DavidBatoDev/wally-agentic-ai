# backend/src/agent/functions/smart_router.py

import json
import os
import requests
from typing import Dict, Any, Optional, List
from src.agent.agent_state import AgentState, WorkflowStatus


class SmartRouter:
    """
    AI-powered router that uses Gemini 2.0 Flash to decide the next node
    based on the current agent state and workflow context.
    Updated for new AgentState structure.
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
    
    def route_next(self, state: AgentState) -> str:
        """
        Use Gemini AI to determine the next node based on agent state.
        
        Args:
            state: Current AgentState containing all workflow information
            
        Returns:
            str: Name of the next node to execute
        """
        try:
            # Check if upload is not templatable and skip template workflow
            if self._should_skip_template_workflow(state):
                print("Smart router: Upload is not templatable, skipping template workflow")
                return "tools"  # Return to tools for user interaction
            
            # Prepare the state summary for the AI
            state_summary = self._prepare_state_summary(state)
            
            # Create the prompt for Gemini
            prompt = self._create_routing_prompt(state_summary)
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            # Extract the node decision
            suggested_node = self._extract_node_decision(response)
            
            # CRITICAL: Validate the suggested node against agent state
            validated_node = self._validate_and_redirect_node(suggested_node, state)
            
            print(f"Smart router: AI suggested '{suggested_node}' -> validated as '{validated_node}'")
            return validated_node
            
        except Exception as e:
            print(f"Smart router error: {e}")
            # Fallback to simple rule-based routing
            return self._fallback_routing(state)
    
    def _should_skip_template_workflow(self, state: AgentState) -> bool:
        """
        Check if we should skip the template workflow because the upload is not templatable.
        """
        if not state.latest_upload:
            return False
        
        # Check if the upload has analysis and is_templatable is False
        if hasattr(state.latest_upload, 'analysis') and state.latest_upload.analysis:
            if hasattr(state.latest_upload.analysis, 'is_templatable'):
                is_templatable = state.latest_upload.analysis.is_templatable
                if not is_templatable:
                    return True
        
        # Fallback to the direct is_templatable attribute if it exists
        if hasattr(state.latest_upload, 'is_templatable'):
            return not state.latest_upload.is_templatable
        
        return False
    
    def _has_sufficient_filled_fields(self, state: AgentState) -> bool:
        """
        Check if filled_fields has at least 3 non-empty key-value pairs.
        """
        filled_fields = state.current_document_in_workflow_state.filled_fields
        if not filled_fields or not isinstance(filled_fields, dict):
            return False
        
        # Count non-empty values
        non_empty_count = 0
        for key, value in filled_fields.items():
            if value and str(value).strip():  # Check if value exists and is not empty/whitespace
                non_empty_count += 1
        
        return non_empty_count >= 3
    
    def _validate_and_redirect_node(self, suggested_node: str, state: AgentState) -> str:
        """
        Validate if the suggested node can actually execute based on agent state.
        If not, redirect to the appropriate node that satisfies the prerequisites.
        """
        # CRITICAL: Check if upload is not templatable before allowing template workflow nodes
        if self._should_skip_template_workflow(state):
            template_workflow_nodes = [
                "ask_user_desired_langauge", "find_template", "extract_required_fields",
                "translate_required_fields", "fill_template", "manual_fill_template"
            ]
            if suggested_node in template_workflow_nodes:
                print(f"Redirecting '{suggested_node}' to 'tools' - upload is not templatable")
                return "tools"
        
        # CRITICAL: Check for sufficient filled fields before translation
        if suggested_node == "translate_required_fields":
            if not self._has_sufficient_filled_fields(state):
                print(f"Redirecting '{suggested_node}' to 'extract_required_fields' - insufficient filled fields")
                return "extract_required_fields"
        
        # Check if the suggested node has all required prerequisites
        if self._can_execute_node(suggested_node, state):
            return suggested_node
        
        # If not, find the correct node that should execute first
        corrected_node = self._find_prerequisite_node(suggested_node, state)
        
        print(f"Node validation: '{suggested_node}' cannot execute, redirecting to '{corrected_node}'")
        return corrected_node
    
    def _can_execute_node(self, node: str, state: AgentState) -> bool:
        """
        Check if a node can execute based on current agent state.
        Updated for new AgentState structure.
        """
        # CRITICAL: Block template workflow nodes if upload is not templatable
        if self._should_skip_template_workflow(state):
            template_workflow_nodes = [
                "ask_user_desired_langauge", "find_template", "extract_required_fields",
                "translate_required_fields", "fill_template", "manual_fill_template"
            ]
            if node in template_workflow_nodes:
                print(f"Node '{node}' blocked - upload is not templatable")
                return False
        
        # CRITICAL: Special validation for translate_required_fields
        if node == "translate_required_fields":
            if not self._has_sufficient_filled_fields(state):
                print(f"Node '{node}' blocked - insufficient filled fields (need at least 3 non-empty)")
                return False
        
        node_requirements = {
            "tools": {
                # Tools can always execute - no prerequisites
                "required_fields": [],
                "description": "Handle user interactions and file uploads"
            },
            "ask_user_desired_langauge": {
                "required_fields": ["latest_upload"],
                "description": "Ask user for desired translation language"
            },
            "find_template": {
                "required_fields": ["latest_upload", "translate_to"],
                "description": "Find matching template"
            },
            "extract_required_fields": {
                "required_fields": ["current_document_template_id", "current_document_template_file_public_url", "current_document_template_required_fields"],
                "description": "Extract fields from document"
            },
            "translate_required_fields": {
                "required_fields": ["current_document_filled_fields", "translate_to"],
                "description": "Translate extracted fields (requires at least 3 non-empty filled fields)"
            },
            "fill_template": {
                "required_fields": ["current_document_template_file_public_url", "current_document_translated_fields"],
                "description": "Generate final document"
            },
            "manual_fill_template": {
                "required_fields": ["current_document_template_required_fields"],
                "description": "Manual template filling"
            },
            "change_specific_field": {
                "required_fields": ["current_document_filled_fields"],
                "description": "Update specific fields"
            },
            "save": {
                # Save can always execute
                "required_fields": [],
                "description": "Save current state"
            },
            "end_node": {
                "required_fields": ["current_document_version_id"],
                "description": "Complete workflow"
            }
        }
        
        if node not in node_requirements:
            print(f"Unknown node: {node}")
            return False
        
        required_fields = node_requirements[node]["required_fields"]
        
        # Check each required field
        for field in required_fields:
            field_value = self._get_field_value(state, field)
            
            # Check if field exists and is not empty
            if field_value is None:
                print(f"Node '{node}' missing required field: {field} (None)")
                return False
            
            # For string fields, check if not empty
            if isinstance(field_value, str) and not field_value.strip():
                print(f"Node '{node}' missing required field: {field} (empty string)")
                return False
            
            # For dict fields, check if not empty
            if isinstance(field_value, dict) and not field_value:
                print(f"Node '{node}' missing required field: {field} (empty dict)")
                return False
            
            # For list fields, check if not empty
            if isinstance(field_value, list) and not field_value:
                print(f"Node '{node}' missing required field: {field} (empty list)")
                return False
        
        print(f"Node '{node}' can execute - all prerequisites satisfied")
        return True
    
    def _get_field_value(self, state: AgentState, field: str) -> Any:
        """
        Get field value from the state, handling nested fields in current_document_in_workflow_state.
        """
        # Handle direct state fields
        if field == "latest_upload":
            return state.latest_upload
        elif field == "translate_to":
            return state.translate_to
        elif field == "current_document_version_id":
            return state.current_document_version_id
        
        # Handle current_document_in_workflow_state nested fields
        elif field.startswith("current_document_"):
            nested_field = field.replace("current_document_", "")
            if hasattr(state.current_document_in_workflow_state, nested_field):
                return getattr(state.current_document_in_workflow_state, nested_field)
            else:
                # Handle field name mappings
                field_mappings = {
                    "template_id": "template_id",
                    "template_file_public_url": "template_file_public_url",
                    "template_required_fields": "template_required_fields",
                    "filled_fields": "filled_fields",
                    "translated_fields": "translated_fields"
                }
                mapped_field = field_mappings.get(nested_field, nested_field)
                return getattr(state.current_document_in_workflow_state, mapped_field, None)
        
        # Fallback to direct attribute access
        return getattr(state, field, None)
    
    def _find_prerequisite_node(self, target_node: str, state: AgentState) -> str:
        """
        Find the node that should execute first to satisfy prerequisites for the target node.
        Updated for new workflow structure.
        """
        # CRITICAL: If upload is not templatable, redirect all template workflow requests to tools
        if self._should_skip_template_workflow(state):
            template_workflow_nodes = [
                "ask_user_desired_langauge", "find_template", "extract_required_fields",
                "translate_required_fields", "fill_template", "manual_fill_template"
            ]
            if target_node in template_workflow_nodes:
                print(f"Prerequisite check: redirecting '{target_node}' to 'tools' - upload not templatable")
                return "tools"
        
        # CRITICAL: If target is translate_required_fields but insufficient filled fields
        if target_node == "translate_required_fields" and not self._has_sufficient_filled_fields(state):
            print(f"Prerequisite check: redirecting '{target_node}' to 'extract_required_fields' - insufficient filled fields")
            return "extract_required_fields"
        
        # Define the workflow dependency chain
        dependency_chain = [
            ("tools", []),  # No prerequisites
            ("ask_user_desired_langauge", ["latest_upload"]),
            ("find_template", ["latest_upload", "translate_to"]),
            ("extract_required_fields", ["current_document_template_id", "current_document_template_file_public_url", "current_document_template_required_fields"]),
            ("translate_required_fields", ["current_document_filled_fields", "translate_to"]),
            ("fill_template", ["current_document_template_file_public_url", "current_document_translated_fields"]),
            ("end_node", ["current_document_version_id"])
        ]
        
        # Special case: Handle tool calls first
        if (state.messages and 
            hasattr(state.messages[-1], "tool_calls") and 
            getattr(state.messages[-1], "tool_calls", None)):
            return "tools"
        
        # Check what's missing and find the earliest node that can satisfy it
        for node, prerequisites in dependency_chain:
            if self._can_execute_node(node, state):
                continue  # This node can already execute, check next
            else:
                # This node cannot execute, so we need to execute it or its prerequisites
                return self._find_immediate_prerequisite(node, state)
        
        # If we reach here, find what the target node specifically needs
        return self._find_immediate_prerequisite(target_node, state)
    
    def _find_immediate_prerequisite(self, node: str, state: AgentState) -> str:
        """
        Find the immediate next node needed to satisfy the given node's prerequisites.
        Updated for new workflow structure.
        """
        # CRITICAL: If upload is not templatable, redirect to tools
        if self._should_skip_template_workflow(state):
            template_workflow_nodes = [
                "ask_user_desired_langauge", "find_template", "extract_required_fields",
                "translate_required_fields", "fill_template", "manual_fill_template"
            ]
            if node in template_workflow_nodes:
                return "tools"
        
        # CRITICAL: If node needs sufficient filled fields but doesn't have them
        if node == "translate_required_fields" and not self._has_sufficient_filled_fields(state):
            return "extract_required_fields"
        
        # Check what's missing for each field and determine the node that provides it
        field_providers = {
            "latest_upload": "tools",
            "translate_to": "ask_user_desired_langauge",
            "current_document_template_id": "find_template",
            "current_document_template_file_public_url": "find_template",
            "current_document_template_required_fields": "find_template",
            "current_document_filled_fields": "extract_required_fields",
            "current_document_translated_fields": "translate_required_fields",
            "current_document_version_id": "fill_template"
        }
        
        node_requirements = {
            "ask_user_desired_langauge": ["latest_upload"],
            "find_template": ["latest_upload", "translate_to"],
            "extract_required_fields": ["current_document_template_id", "current_document_template_file_public_url", "current_document_template_required_fields"],
            "translate_required_fields": ["current_document_filled_fields", "translate_to"],
            "fill_template": ["current_document_template_file_public_url", "current_document_translated_fields"],
            "end_node": ["current_document_version_id"]
        }
        
        if node not in node_requirements:
            return "tools"  # Default fallback
        
        # Find the first missing required field
        for required_field in node_requirements[node]:
            field_value = self._get_field_value(state, required_field)
            
            # Check if field is missing or empty
            is_missing = (
                field_value is None or
                (isinstance(field_value, str) and not field_value.strip()) or
                (isinstance(field_value, dict) and not field_value) or
                (isinstance(field_value, list) and not field_value)
            )
            
            if is_missing:
                provider_node = field_providers.get(required_field, "tools")
                print(f"Missing field '{required_field}' for node '{node}', need to execute '{provider_node}'")
                
                # Recursively check if the provider node can execute
                if self._can_execute_node(provider_node, state):
                    return provider_node
                else:
                    # Provider node also can't execute, find its prerequisite
                    return self._find_immediate_prerequisite(provider_node, state)
        
        # If we reach here, the node should be able to execute
        return node
    
    def _prepare_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        Create a clean summary of the agent state for the AI to analyze.
        Updated for new AgentState structure.
        """
        current_doc = state.current_document_in_workflow_state
        latest_upload = state.latest_upload
        
        # Enhanced templatable check
        upload_is_templatable = False
        if latest_upload:
            if hasattr(latest_upload, 'analysis') and latest_upload.analysis:
                if hasattr(latest_upload.analysis, 'is_templatable'):
                    upload_is_templatable = latest_upload.analysis.is_templatable
            elif hasattr(latest_upload, 'is_templatable'):
                upload_is_templatable = latest_upload.is_templatable
        
        # Check filled fields sufficiency
        has_sufficient_filled_fields = self._has_sufficient_filled_fields(state)
        filled_fields_count = 0
        if current_doc.filled_fields:
            filled_fields_count = sum(1 for v in current_doc.filled_fields.values() if v and str(v).strip())
        
        return {
            "workflow_status": state.workflow_status.value,
            "steps_done": state.steps_done,
            "has_upload": latest_upload is not None,
            "upload_is_templatable": upload_is_templatable,
            "has_translate_to": bool(state.translate_to),
            "translate_to": state.translate_to,
            "has_template": bool(current_doc.template_id),
            "has_template_fields": bool(current_doc.template_required_fields),
            "has_extracted_fields": bool(current_doc.extracted_fields_from_raw_ocr),
            "has_filled_fields": bool(current_doc.filled_fields),
            "has_sufficient_filled_fields": has_sufficient_filled_fields,
            "filled_fields_count": filled_fields_count,
            "has_translated_fields": bool(current_doc.translated_fields),
            "has_document_output": bool(state.current_document_version_id),
            "last_message_has_tool_calls": (
                bool(state.messages) and 
                hasattr(state.messages[-1], 'tool_calls') and 
                bool(getattr(state.messages[-1], 'tool_calls', None))
            ),
            "recent_messages_count": len(state.messages),
            "template_required_fields": list(current_doc.template_required_fields.keys()) if current_doc.template_required_fields else [],
            "extracted_fields_keys": list(current_doc.extracted_fields_from_raw_ocr.keys()) if current_doc.extracted_fields_from_raw_ocr else [],
            "filled_fields_keys": list(current_doc.filled_fields.keys()) if current_doc.filled_fields else [],
            "translated_fields_keys": list(current_doc.translated_fields.keys()) if current_doc.translated_fields else [],
            "context_flags": state.context,
            "current_pending_node": state.current_pending_node,
            "conversation_analysis": self._analyze_conversation(state.messages)
        }
    
    def _analyze_conversation(self, messages: List[Any]) -> Dict[str, Any]:
        """
        Analyze the conversation history to understand user intent and current needs.
        """
        if not messages:
            return {"summary": "No conversation history", "user_intent": "unknown", "last_user_action": None}
        
        # Get recent messages (last 10 to avoid token limits)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        conversation_text = []
        user_messages = []
        ai_messages = []
        tool_messages = []
        
        for msg in recent_messages:
            msg_type = type(msg).__name__
            content = getattr(msg, 'content', '')
            
            if 'Human' in msg_type or 'User' in msg_type:
                user_messages.append(content)
                conversation_text.append(f"USER: {content}")
            elif 'AI' in msg_type or 'Assistant' in msg_type:
                ai_messages.append(content)
                conversation_text.append(f"AI: {content}")
            elif 'Tool' in msg_type:
                tool_messages.append(content)
                conversation_text.append(f"TOOL: {content}")
        
        # Analyze user intent from their messages
        user_intent = self._detect_user_intent(user_messages)
        
        # Get the last user action/request
        last_user_message = user_messages[-1] if user_messages else None
        last_user_action = self._categorize_user_action(last_user_message) if last_user_message else None
        
        return {
            "conversation_summary": " | ".join(conversation_text),
            "user_intent": user_intent,
            "last_user_action": last_user_action,
            "total_user_messages": len(user_messages),
            "total_ai_messages": len(ai_messages),
            "has_tool_activity": len(tool_messages) > 0,
            "last_user_message": last_user_message,
            "conversation_length": len(recent_messages)
        }
    
    def _detect_user_intent(self, user_messages: List[str]) -> str:
        """
        Detect the overall user intent from their messages.
        """
        if not user_messages:
            return "unknown"
        
        # Combine all user messages to analyze intent
        combined_text = " ".join(user_messages).lower()
        
        # Define intent patterns
        intent_patterns = {
            "translate_document": ["translate", "translation", "convert", "language"],
            "upload_file": ["upload", "file", "document", "attach"],
            "correct_field": ["correct", "fix", "change", "wrong", "mistake", "update"],
            "confirm_action": ["yes", "ok", "continue", "proceed", "confirm"],
            "request_help": ["help", "what", "how", "explain"],
            "cancel_workflow": ["cancel", "stop", "abort", "quit"],
            "download_result": ["download", "get", "result", "final"],
            "start_over": ["restart", "start over", "begin again", "new"],
            "specify_language": ["spanish", "french", "german", "chinese", "japanese", "korean", "tagalog", "filipino"]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the highest scoring intent, or "general_interaction" if none match
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return "general_interaction"
    
    def _categorize_user_action(self, last_message: str) -> str:
        """
        Categorize the user's last action/message.
        """
        if not last_message:
            return "no_action"
        
        message = last_message.lower().strip()
        
        # Direct action indicators
        if any(word in message for word in ["upload", "attach", "file"]):
            return "wants_to_upload"
        elif any(word in message for word in ["yes", "ok", "continue", "proceed"]):
            return "confirms_action"
        elif any(word in message for word in ["no", "cancel", "stop"]):
            return "rejects_action"
        elif any(word in message for word in ["correct", "fix", "change", "wrong"]):
            return "requests_correction"
        elif any(word in message for word in ["help", "what", "how", "explain"]):
            return "asks_question"
        elif any(word in message for word in ["download", "get", "result"]):
            return "requests_download"
        elif any(word in message for word in ["spanish", "french", "german", "chinese", "japanese", "korean", "tagalog", "filipino"]):
            return "specifies_language"
        elif len(message.split()) > 10:  # Long message likely contains information
            return "provides_information"
        else:
            return "general_response"
    
    def _create_routing_prompt(self, state_summary: Dict[str, Any]) -> str:
        """
        Create a structured prompt for Gemini to decide the next workflow node.
        Updated for new workflow structure.
        """
        conversation_analysis = state_summary.get("conversation_analysis", {})
        
        return f"""You are a smart router for a document translation workflow. Based on the current state and conversation history, decide what the next node should be.

CURRENT WORKFLOW STATE:
- Status: {state_summary['workflow_status']}
- Steps completed: {state_summary['steps_done']}
- Has uploaded file: {state_summary['has_upload']}
- Upload is templatable: {state_summary['upload_is_templatable']}
- Has target language: {state_summary['has_translate_to']}
- Target language: {state_summary['translate_to']}
- Has template: {state_summary['has_template']}
- Template fields defined: {state_summary['has_template_fields']}
- Fields extracted: {state_summary['has_extracted_fields']}
- Fields filled: {state_summary['has_filled_fields']}
- Has sufficient filled fields (≥3): {state_summary['has_sufficient_filled_fields']}
- Filled fields count: {state_summary['filled_fields_count']}
- Fields translated: {state_summary['has_translated_fields']}
- Final document ready: {state_summary['has_document_output']}
- Last message has tool calls: {state_summary['last_message_has_tool_calls']}
- Current pending node: {state_summary['current_pending_node']}

CONVERSATION ANALYSIS:
- User intent: {conversation_analysis.get('user_intent', 'unknown')}
- Last user action: {conversation_analysis.get('last_user_action', 'unknown')}
- Conversation length: {conversation_analysis.get('conversation_length', 0)} messages
- Last user message: "{conversation_analysis.get('last_user_message', 'None')}"
- Recent conversation: {conversation_analysis.get('conversation_summary', 'No conversation')}

AVAILABLE NODES:
- tools: Execute tool calls or handle user interactions (always available)
- ask_user_desired_langauge: Ask user for desired translation language (requires uploaded file AND file must be templatable)
- find_template: Find matching template (requires uploaded file and target language)
- extract_required_fields: Extract text fields from document (requires template)
- translate_required_fields: Translate extracted fields (requires at least 3 non-empty filled fields and target language)
- fill_template: Generate final document (requires translated fields)
- manual_fill_template: Manual template filling (requires template fields)
- change_specific_field: Update specific fields based on user corrections
- save: Save current state to database (always available)
- end_node: Complete the workflow (requires final document)

CRITICAL RULES:
1. If upload is not templatable, NEVER route to template workflow nodes (ask_user_desired_langauge, find_template, extract_required_fields, translate_required_fields, fill_template, manual_fill_template). Always use 'tools' instead.
2. NEVER route to 'translate_required_fields' unless there are at least 3 non-empty filled fields.
3. Only use nodes that exist in your LangGraph workflow.

INTELLIGENT ROUTING LOGIC:
Consider both the workflow state AND the conversation context:

1. PRIORITY: Tool calls in last message → tools
2. File uploaded but NOT TEMPLATABLE → tools (inform user about non-templatable file)
3. File uploaded, IS TEMPLATABLE, but no target language → ask_user_desired_langauge
4. User specifies language AND file is templatable → ask_user_desired_langauge (to process the language)
5. User wants to upload file OR no file uploaded yet → tools
6. File uploaded, language specified, templatable, but no template → find_template
7. Template found but fields not extracted/filled → extract_required_fields
8. Fields filled but insufficient (< 3 non-empty) → extract_required_fields (continue extraction)
9. User requests corrections → change_specific_field
10. Fields filled sufficiently (≥ 3 non-empty fields), translation needed → translate_required_fields
11. Ready for final document → fill_template
12. User asks questions or needs help → tools
13. Workflow complete → end_node
14. Default/save state → save

CONTEXTUAL CONSIDERATIONS:
- If user says "yes/ok/continue" and workflow can proceed → next logical step
- If user provides corrections → change_specific_field
- If user specifies a language BUT file is not templatable → tools
- Match the user's current intent with appropriate workflow step
- Consider what prerequisites are satisfied
- ALWAYS check if upload is templatable before proceeding with template workflow
- ALWAYS check if sufficient filled fields exist before translation

RESPONSE FORMAT:
Return ONLY the node name as a single word. No explanations, no quotes, no additional text.

Next node:"""
    
    def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """
        Make API call to Gemini 2.0 Flash.
        """
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for consistent routing decisions
                "maxOutputTokens": 10,  # We only need one word
                "topP": 0.8,
                "topK": 40
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()
    
    def _extract_node_decision(self, api_response: Dict[str, Any]) -> str:
        """
        Extract the node decision from Gemini's response.
        """
        try:
            # Navigate the Gemini response structure
            candidates = api_response.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in API response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ValueError("No parts in response content")
            
            text = parts[0].get("text", "").strip()
            
            # Validate the node name
            valid_nodes = [
                "ask_user_desired_langauge", "find_template", "extract_required_fields",
                "translate_required_fields", "fill_template", "manual_fill_template",
                "change_specific_field", "tools", "save", "end_node"
            ]
            
            if text in valid_nodes:
                return text
            else:
                print(f"Invalid node returned by AI: {text}")
                return "save"  # Safe fallback
                
        except Exception as e:
            print(f"Error extracting node decision: {e}")
            return "save"  # Safe fallback
    
    def _fallback_routing(self, state: AgentState) -> str:
        """
        Fallback to rule-based routing if AI fails.
        Updated for new workflow structure.
        """
        # CRITICAL: Check if upload is not templatable first
        if self._should_skip_template_workflow(state):
            return "tools"
        
        # Check tool calls first
        if (state.messages and 
            hasattr(state.messages[-1], "tool_calls") and 
            getattr(state.messages[-1], "tool_calls", None)):
            return "tools"
        
        # Check for pending node
        if state.current_pending_node:
            return state.current_pending_node
        
        # Check if we have an upload but no language specified (and upload is templatable)
        if state.latest_upload and not state.translate_to:
            return "ask_user_desired_langauge"
        
        # Check if we have upload and language but no template
        if (state.latest_upload and state.translate_to and 
            not state.current_document_in_workflow_state.template_id):
            return "find_template"
        
        # Check if we have template but no extracted fields
        if (state.current_document_in_workflow_state.template_id and 
            not state.current_document_in_workflow_state.filled_fields):
            return "extract_required_fields"
        
        # Default fallback
        return "save"


# Convenience function to use the smart router
def route_next(state: AgentState) -> str:
    """
    Main routing function that uses the SmartRouter.
    This can be used as a drop-in replacement for your existing route_next method.
    """
    router = SmartRouter()
    return router.route_next(state)