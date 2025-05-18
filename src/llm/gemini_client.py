"""
Gemini LLM client for interacting with Google's Gemini API.
"""

import json
from typing import Dict, List, Any, Optional, Union, Tuple
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

settings = get_settings()

# Configure the Gemini client
genai.configure(api_key=settings.GEMINI_API_KEY)

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]


class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini client.
        
        Args:
            model_name: The name of the Gemini model to use
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=DEFAULT_GENERATION_CONFIG,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt for the model
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text
        """
        generation_config = DEFAULT_GENERATION_CONFIG.copy()
        
        if temperature is not None:
            generation_config["temperature"] = temperature
        
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            chat = self.model.start_chat()
            
            if system_prompt:
                chat = self.model.start_chat(history=[
                    {"role": "user", "parts": ["You are a helpful assistant."]},
                    {"role": "model", "parts": [system_prompt]},
                ])
            
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Error in generate_text: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_for_agent(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response for an agent with tool calling.
        
        Args:
            prompt: The prompt to generate from
            tools: List of tool definitions
            chat_history: Optional chat history
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with the response text and tool calls
        """
        generation_config = DEFAULT_GENERATION_CONFIG.copy()
        
        try:
            # Prepare the prompt with tools
            full_prompt = ""
            
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            
            full_prompt += "Available Tools:\n"
            for tool in tools:
                full_prompt += f"- {tool['name']}: {tool['description']}\n"
                full_prompt += f"  Parameters: {json.dumps(tool['parameters'])}\n\n"
            
            full_prompt += "\nWhen you want to use a tool, respond in the format:\n"
            full_prompt += '{"thought": "your reasoning", "tool": "tool_name", "tool_input": {"param1": "value1", ...}}\n\n'
            
            if chat_history:
                full_prompt += "Previous conversation:\n"
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    full_prompt += f"{role}: {content}\n"
                
                full_prompt += "\n"
            
            full_prompt += f"User request: {prompt}\n\n"
            full_prompt += "Your response (remember to use the tool format when appropriate):"
            
            response = await self.generate_text(full_prompt)
            
            # Parse the response to see if it's a tool call
            tool_call = None
            try:
                if "{" in response and "}" in response:
                    # Extract JSON object from response if it exists
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_str = response[json_start:json_end]
                    
                    tool_call = json.loads(json_str)
                    
                    # Validate tool call format
                    if "tool" in tool_call and "tool_input" in tool_call:
                        return {
                            "is_tool_call": True,
                            "tool": tool_call["tool"],
                            "tool_input": tool_call["tool_input"],
                            "thought": tool_call.get("thought", ""),
                            "original_text": response
                        }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error parsing tool call: {e}")
            
            # If not a valid tool call, return as normal text
            return {
                "is_tool_call": False,
                "text": response,
                "original_text": response
            }
        except Exception as e:
            print(f"Error in generate_for_agent: {e}")
            raise
    
    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text using the Gemini embedding model.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding floats
        """
        embedding_model = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        
        return embedding_model["embedding"]


# Singleton instance
gemini_client = GeminiClient()