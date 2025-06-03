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
            # Prepare the state summary for the AI
            state_summary = self._prepare_state_summary(state)
            
            # Create the prompt for Gemini
            prompt = self._create_routing_prompt(state_summary)
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            # Extract and validate the node decision
            next_node = self._extract_node_decision(response)
            
            return next_node
            
        except Exception as e:
            print(f"Smart router error: {e}")
            # Fallback to simple rule-based routing
            return self._fallback_routing(state)
    
    def _prepare_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        Create a clean summary of the agent state for the AI to analyze.
        """
        return {
            "workflow_status": state.workflow_status.value,
            "steps_done": state.steps_done,
            "has_upload": bool(state.user_upload_id),
            "upload_analyzed": state.upload_analysis is not None,
            "has_template": bool(state.template_id),
            "has_extracted_fields": bool(state.extracted_required_fields),
            "has_translated_fields": bool(state.translated_required_fields),
            "translate_to": state.translate_to,
            "has_document_output": bool(state.document_version_id),
            "last_message_has_tool_calls": (
                bool(state.messages) and 
                hasattr(state.messages[-1], 'tool_calls') and 
                bool(getattr(state.messages[-1], 'tool_calls', None))
            ),
            "recent_messages_count": len(state.messages),
            "template_required_fields": list(state.template_required_fields.keys()) if state.template_required_fields else [],
            "extracted_fields_keys": list(state.extracted_required_fields.keys()) if state.extracted_required_fields else [],
            "context_flags": state.context,
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
            "start_over": ["restart", "start over", "begin again", "new"]
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
        elif len(message.split()) > 10:  # Long message likely contains information
            return "provides_information"
        else:
            return "general_response"
    
    def _create_routing_prompt(self, state_summary: Dict[str, Any]) -> str:
        """
        Create a structured prompt for Gemini to decide the next workflow node.
        """
        conversation_analysis = state_summary.get("conversation_analysis", {})
        
        return f"""You are a smart router for a document translation workflow. Based on the current state and conversation history, decide what the next node should be.

CURRENT WORKFLOW STATE:
- Status: {state_summary['workflow_status']}
- Steps completed: {state_summary['steps_done']}
- Has uploaded file: {state_summary['has_upload']}
- Upload analyzed: {state_summary['upload_analyzed']}
- Has template: {state_summary['has_template']}
- Fields extracted: {state_summary['has_extracted_fields']}
- Fields translated: {state_summary['has_translated_fields']}
- Target language: {state_summary['translate_to']}
- Final document ready: {state_summary['has_document_output']}
- Last message has tool calls: {state_summary['last_message_has_tool_calls']}

CONVERSATION ANALYSIS:
- User intent: {conversation_analysis.get('user_intent', 'unknown')}
- Last user action: {conversation_analysis.get('last_user_action', 'unknown')}
- Conversation length: {conversation_analysis.get('conversation_length', 0)} messages
- Last user message: "{conversation_analysis.get('last_user_message', 'None')}"
- Recent conversation: {conversation_analysis.get('conversation_summary', 'No conversation')}

AVAILABLE NODES:
- analyze_doc: Analyze uploaded document if the latest message include uploaded document to detect type, language, and extract metadata only run this if the past few latest message has "[FILE_UPLOADED] and Upload analyzed is not empty"
- find_template_node: Find matching template based on document analysis
- extract_required_fields: Extract text fields from the uploaded document using OCR+LLM
- translate_required_fields: Translate extracted fields to target language if Fields
- fill_template: Generate final document by filling template with translated fields
- change_specific_field: Update specific fields based on user corrections
- tools: Execute tool calls from the last AI message
- save: Save current state to database (if user latest message doenst required any node calls)
- end_node: Complete the workflow

INTELLIGENT ROUTING LOGIC:
Consider both the workflow state AND the conversation context:

1. PRIORITY: Tool calls → tools
2. File uploaded but not analyzed → analyze_doc
3. User wants to upload file OR no file uploaded yet -> tools
4. Analysis done but no template → find_template_node  
5. Template found but fields not extracted or extracted fields is empty → extract_required_fields
6. User requests corrections → change_specific_field
7. Fields extracted, translation needed → translate_required_fields
8. Ready for final document → fill_template
9. User asks questions or needs help (select appropriate node)
10. Workflow complete → end_node
11. Default/save state → save

CONTEXTUAL CONSIDERATIONS:
- If user says "yes/ok/continue" and workflow can proceed → next logical step
- If user provides corrections to the extracted_fields → change_specific_field
- Match the user's current intent with appropriate workflow step

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
                "analyze_doc", "find_template_node", "extract_required_fields",
                "translate_required_fields", "fill_template",
                "call_ui_tools", "change_specific_field", "tools",
                "response_node", "save", "end_node"
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
        This mirrors your original route_next logic.
        """
        if state.user_upload_id and "analyze_doc" not in state.steps_done:
            return "analyze_doc"
       
        if state.messages and hasattr(state.messages[-1], "tool_calls") and getattr(state.messages[-1], "tool_calls", None):
            return "tools"
       
        return "save"


# Convenience function to use the smart router
def route_next(state: AgentState) -> str:
    """
    Main routing function that uses the SmartRouter.
    This can be used as a drop-in replacement for your existing route_next method.
    """
    router = SmartRouter()
    return router.route_next(state)