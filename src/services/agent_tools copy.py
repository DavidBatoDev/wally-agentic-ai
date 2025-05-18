# backend/src/services/agent_tools.py
"""
Tools registry and implementations for the agent orchestrator.
"""

import json
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from uuid import UUID
import asyncio

from src.models.tool_schema import (
    ToolDescription,
    OCRToolInput, OCRToolOutput,
    TranslateToolInput, TranslateToolOutput,
    ExtractFieldsToolInput, ExtractFieldsToolOutput,
    SearchMemoryToolInput, SearchMemoryToolOutput
)
from utils.db_client import supabase_client


class ToolRegistry:
    """Registry for tools that can be called by the agent."""
    
    def __init__(self):
        """Initialize the tool registry with available tools."""
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # OCR Tool
        self.register_tool(
            name="run_ocr_on_image",
            description="Extract text from an image or document using OCR.",
            handler=self._handle_ocr,
            input_schema=OCRToolInput.schema(),
            output_schema=OCRToolOutput.schema()
        )
        
        # Translation Tool
        self.register_tool(
            name="translate_text",
            description="Translate text from one language to another.",
            handler=self._handle_translation,
            input_schema=TranslateToolInput.schema(),
            output_schema=TranslateToolOutput.schema()
        )
        
        # Field Extraction Tool
        self.register_tool(
            name="extract_fields_from_document",
            description="Extract structured fields from document text using templates.",
            handler=self._handle_field_extraction,
            input_schema=ExtractFieldsToolInput.schema(),
            output_schema=ExtractFieldsToolOutput.schema()
        )
        
        # Memory Search Tool
        self.register_tool(
            name="search_conversation_memory",
            description="Search for relevant information in the conversation history.",
            handler=self._handle_memory_search,
            input_schema=SearchMemoryToolInput.schema(),
            output_schema=SearchMemoryToolOutput.schema()
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        return_direct: bool = False
    ) -> None:
        """
        Register a new tool in the registry.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Async function to handle the tool call
            input_schema: JSON schema for tool inputs
            output_schema: JSON schema for tool outputs
            return_direct: Whether to return the tool output directly
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "return_direct": return_direct
        }
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tool definitions
        """
        return self._tools
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get formatted tool descriptions for LLM prompt.
        
        Returns:
            List of tool descriptions
        """
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append({
                "name": name,
                "description": tool["description"],
                "parameters": tool["input_schema"]["properties"]
            })
        return descriptions
    
    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        conversation_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given inputs.
        
        Args:
            tool_name: The name of the tool to execute
            tool_input: Input parameters for the tool
            conversation_id: Optional conversation ID for context
            context: Optional additional context
            
        Returns:
            Tool execution result
        """
        context = context or {}
        if conversation_id:
            context["conversation_id"] = str(conversation_id)
            
        if tool_name not in self._tools:
            return {
                "error": f"Tool '{tool_name}' not found",
                "status": "error"
            }
        
        try:
            tool = self._tools[tool_name]
            handler = tool["handler"]
            
            # Execute the tool handler
            result = await handler(tool_input, context)
            return {
                "status": "success",
                "data": result
            }
        except Exception as e:
            print(f"Error executing tool '{tool_name}': {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "status": "error"
            }
    
    # Tool Handlers
    
    async def _handle_ocr(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle OCR requests.
        
        Args:
            input_data: Tool input parameters
            context: Optional execution context
            
        Returns:
            OCR results
        """
        try:
            # Validate input
            ocr_input = OCRToolInput(**input_data)
            file_id = ocr_input.file_id
            
            # In a real implementation, you would:
            # 1. Retrieve the file from Supabase Storage
            # 2. Call Google Cloud Vision or similar OCR service
            # 3. Process the OCR results
            
            # Simulated OCR result for demonstration
            # In reality, you would make API calls to OCR service here
            await asyncio.sleep(1)  # Simulate API call
            
            return {
                "text": "This is a birth certificate for Juan Dela Cruz, born on January 15, 1985 in Manila, Philippines.",
                "confidence": 0.95
            }
        except Exception as e:
            print(f"OCR error: {e}")
            traceback.print_exc()
            raise
    
    async def _handle_translation(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle translation requests.
        
        Args:
            input_data: Tool input parameters
            context: Optional execution context
            
        Returns:
            Translation results
        """
        try:
            # Validate input
            translate_input = TranslateToolInput(**input_data)
            
            # In a real implementation, you would:
            # 1. Call a translation API (Google Cloud Translation, DeepL, etc.)
            # 2. Return the translated text
            
            # Simulated translation result
            await asyncio.sleep(1)  # Simulate API call
            
            # Example logic for basic demonstration
            source_lang = translate_input.source_language or "tl"  # Default to Tagalog if not specified
            target_lang = translate_input.target_language
            
            return {
                "translated_text": f"[Translation of '{translate_input.text}' from {source_lang} to {target_lang}]",
                "detected_source_language": source_lang
            }
        except Exception as e:
            print(f"Translation error: {e}")
            traceback.print_exc()
            raise
    
    async def _handle_field_extraction(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle document field extraction.
        
        Args:
            input_data: Tool input parameters
            context: Optional execution context
            
        Returns:
            Extracted fields
        """
        try:
            # Validate input
            extract_input = ExtractFieldsToolInput(**input_data)
            
            # In a real implementation, you would:
            # 1. Look up the template definition (from Supabase or memory)
            # 2. Use regex patterns or LLM to extract fields
            # 3. Return structured data
            
            # Simulated extraction result
            await asyncio.sleep(1)  # Simulate processing
            
            # Determine document type based on content
            text = extract_input.text.lower()
            doc_type = extract_input.doc_type or "unknown"
            
            # Simple field extraction logic (would be more sophisticated in real app)
            fields = {}
            missing_fields = []
            
            if "birth certificate" in text:
                # Extract basic fields from text using simple logic
                # In a real app, this would use more sophisticated NLP/regex
                if "juan" in text and "dela cruz" in text:
                    fields["full_name"] = "Juan Dela Cruz"
                else:
                    missing_fields.append("full_name")
                
                if "january 15, 1985" in text:
                    fields["birth_date"] = "1985-01-15"
                else:
                    missing_fields.append("birth_date")
                
                if "manila" in text:
                    fields["birth_place"] = "Manila, Philippines"
                else:
                    missing_fields.append("birth_place")
                
                # These fields would typically be extracted by more sophisticated logic
                missing_fields.extend(["mother_name", "father_name"])
            
            return {
                "fields": fields,
                "missing_fields": missing_fields,
                "document_type": doc_type
            }
        except Exception as e:
            print(f"Field extraction error: {e}")
            traceback.print_exc()
            raise
    
    async def _handle_memory_search(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search conversation memory.
        
        Args:
            input_data: Tool input parameters
            context: Optional execution context
            
        Returns:
            Search results
        """
        try:
            # Validate input
            search_input = SearchMemoryToolInput(**input_data)
            
            # In a real implementation, you would:
            # 1. Generate embedding for the query
            # 2. Perform vector search in Supabase (or other vector DB)
            # 3. Return similar memories
            
            # Simulated memory search
            await asyncio.sleep(0.5)  # Simulate search
            
            # Example of how you'd integrate with the Supabase client
            # In this example, we're assuming the embeddings are generated elsewhere
            if context and "conversation_id" in context:
                # results = supabase_client.search_memory(
                #     conversation_id=context["conversation_id"],
                #     query=search_input.query,
                #     limit=search_input.limit
                # )
                
                # Simulated results for demonstration
                results = [
                    {
                        "id": "1234",
                        "content": "User previously mentioned they have a birth certificate from the Philippines.",
                        "similarity": 0.92
                    },
                    {
                        "id": "2345",
                        "content": "User said their document was issued in 1985.",
                        "similarity": 0.85
                    }
                ]
            else:
                results = []
            
            return {
                "results": results,
                "query": search_input.query
            }
        except Exception as e:
            print(f"Memory search error: {e}")
            traceback.print_exc()
            raise


# Create available tools registry
def get_available_tools() -> List[ToolDescription]:
    """
    Get all available tools as descriptions.
    
    Returns:
        List of tool descriptions
    """
    registry = ToolRegistry()
    descriptions = []
    
    for name, tool in registry.get_tools().items():
        descriptions.append(
            ToolDescription(
                name=name,
                description=tool["description"],
                input_schema=tool["input_schema"],
                output_schema=tool["output_schema"]
            )
        )
    
    return descriptions