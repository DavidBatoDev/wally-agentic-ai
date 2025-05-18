# backend/src/services/agent_tools.py
"""
Agent tools registry and implementations for Wally-Chat.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from uuid import UUID
import re
import traceback

# Import our new next_action_tool
from src.services.next_action_tool import register_next_actions_tool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Registry for tools that can be used by the LLM agent.
    Provides registration, discovery, and execution of tools.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[[Dict[str, Any], UUID, Dict[str, Any]], Awaitable[Any]],
        parameters: Dict[str, Any],
        return_direct: bool = False
    ) -> None:
        """
        Register a tool with the registry.
        
        Args:
            name: The name of the tool
            description: Description of what the tool does
            handler: Async function that implements the tool
            parameters: JSON schema describing the expected parameters
            return_direct: Whether to return the result directly without LLM post-processing
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters,
            "input_schema": parameters,
            "return_direct": return_direct
        }
        logger.info(f"Registered tool: {name}")
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tools with their metadata
        """
        return self.tools
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all tools in a format suitable for LLM.
        
        Returns:
            List of tool descriptions with name, description, and parameters
        """
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append({
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        return tool_descriptions
    
    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        conversation_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a tool by name with the given input.
        
        Args:
            tool_name: The name of the tool to execute
            tool_input: The input parameters for the tool
            conversation_id: Optional conversation ID for context
            context: Additional context for tool execution
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool doesn't exist
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        context = context or {}
        
        try:
            logger.info(f"Executing tool: {tool_name} with input: {json.dumps(tool_input)}")
            result = await tool["handler"](tool_input, conversation_id, context)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            traceback.print_exc()
            return {"error": str(e), "status": "error"}
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        self._register_calculator_tool()
        # Register the next actions tool
        register_next_actions_tool(self)
        # Add more default tools here as needed
    
    def _register_calculator_tool(self) -> None:
        """Register the calculator tool."""
        async def calculator_handler(
            tool_input: Dict[str, Any],
            conversation_id: Optional[UUID] = None,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Execute math calculations based on the provided expression.
            
            Args:
                tool_input: Dictionary containing the expression to evaluate
                conversation_id: Optional conversation ID
                context: Additional context
                
            Returns:
                Result of the calculation
            """
            expression = tool_input.get("expression", "")
            
            # Clean and validate the expression for security
            # Only allow basic arithmetic operations and common math functions
            cleaned_expr = self._sanitize_math_expression(expression)
            
            try:
                # Use a safer approach to evaluate math expressions
                # Note: In a production environment, consider using a library like 'numexpr'
                import math
                # Define allowed globals with only math functions and constants
                allowed_globals = {
                    name: getattr(math, name)
                    for name in dir(math)
                    if not name.startswith('_')
                }
                # Add basic operations
                allowed_globals.update({
                    'abs': abs,
                    'round': round,
                    'min': min,
                    'max': max,
                })
                
                # Evaluate the expression with restricted globals and no locals
                result = eval(cleaned_expr, {"__builtins__": {}}, allowed_globals)
                
                return {
                    "status": "success",
                    "expression": expression,
                    "result": result,
                    "formatted_result": str(result)
                }
            
            except Exception as e:
                logger.error(f"Error evaluating expression '{expression}': {str(e)}")
                return {
                    "status": "error",
                    "expression": expression,
                    "error": f"Unable to evaluate expression: {str(e)}"
                }
        
        self.register_tool(
            name="calculator",
            description="Evaluates mathematical expressions. Can handle basic arithmetic, trigonometric functions, logarithms, etc.",
            handler=calculator_handler,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sin(0.5)', 'log(100)')"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def _sanitize_math_expression(self, expression: str) -> str:
        """
        Sanitize a math expression to prevent code injection.
        
        Args:
            expression: The expression to sanitize
            
        Returns:
            Sanitized expression
            
        Raises:
            ValueError: If the expression contains disallowed characters or patterns
        """
        # Remove whitespace
        expression = expression.strip()
        
        # Check for disallowed patterns (anything that might be used for code execution)
        disallowed_patterns = [
            r'__.*__',  # Dunder methods
            r'import',  # Import statements
            r'eval',    # Eval function
            r'exec',    # Exec function
            r'compile', # Compile function
            r'open',    # File operations
            r'os\.',    # OS module
            r'sys\.',   # Sys module
            r'subprocess', # Subprocess module
            r'lambda',  # Lambda functions
            r'def ',    # Function definitions
            r'class ',  # Class definitions
            r';',       # Multiple statements
        ]
        
        for pattern in disallowed_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Expression contains disallowed pattern: {pattern}")
        
        # Restrict to math operations and simple functions
        allowed_chars = r'[0-9\+\-\*\/\(\)\.\,\s\^\%]'
        allowed_funcs = r'(sin|cos|tan|asin|acos|atan|exp|log|log10|log2|sqrt|pow|abs|round|min|max|floor|ceil)'
        allowed_pattern = f'^({allowed_funcs}|{allowed_chars})+$'
        
        if not re.match(allowed_pattern, expression, re.IGNORECASE):
            raise ValueError("Expression contains disallowed characters or functions")
        
        return expression