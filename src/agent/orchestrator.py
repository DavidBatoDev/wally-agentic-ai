# backend/src/agent/orchestrator.py
from typing import Dict, Any, List, Optional
import json
import uuid
from langchain.agents import AgentExecutor, create_structured_chat_agent, AgentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel
from src.agent.agent_tools import get_tools

class AgentOrchestrator:
    """
    Agent orchestrator that processes user messages and manages tools.
    It handles conversation state and produces structured responses using LCEL.
    """
    
    def __init__(self, llm, tools: Optional[List[BaseTool]] = None, db_client=None):
        """
        Initialize the AgentOrchestrator.
        
        Args:
            llm: The LLM instance from Google's Gemini API or similar
            tools: List of LangChain tools the agent can use (if None, tools will be created)
            db_client: Database client for interacting with Supabase
        """
        self.llm = llm
        self.db_client = db_client
        
        # Get tools if not provided
        if tools is None:
            self.tools = get_tools(db_client=db_client)
        else:
            self.tools = tools
        
        # Create the agent chain using LCEL
        self.chain = self._create_agent_chain()
        
    def _create_agent_chain(self):
        """Create a structured chat agent chain using LCEL."""
        # Create system prompt that instructs the agent to return structured outputs
        system_prompt = """You are an intelligent assistant that helps users with various tasks.
        You have access to several tools that can help you fulfill user requests.
        
        Important: Always return your final response in one of these JSON formats:
        
        1. For simple text responses:
        ```json
        {"kind": "text", "text": "Your message here"}
        ```
        
        2. For showing buttons:
        ```json
        {"kind": "buttons", "prompt": "What would you like to do next?", "buttons": [{"label": "Option 1", "action": "action_1"}, {"label": "Option 2", "action": "action_2"}]}
        ```
        
        3. For requesting structured input:
        ```json
        {"kind": "inputs", "prompt": "Please provide the following information:", "inputs": [{"name": "field_1", "label": "Field 1", "type": "text"}, {"name": "field_2", "label": "Field 2", "type": "select", "options": ["option1", "option2"]}], "submit_label": "Submit"}
        ```
        
        4. For file uploads:
        ```json
        {"kind": "buttons", "prompt": "Please upload a file", "buttons": [{"label": "Upload PDF", "action": "upload_file", "file_types": ["application/pdf"]}]}
        ```
        
        5. For file previews:
        ```json
        {"kind": "file_card", "file_id": "file_id_here", "title": "Document Title", "summary": "Brief summary of the document", "status": "ready", "actions": [{"label": "Extract Data", "action": "extract_data"}]}
        ```
        
        Always think carefully about which response format is most appropriate for the current interaction.
        Use the most suitable format based on the user's needs.
        """
        
        # Create the chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the structured chat agent
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        # Create input preparation function
        def prepare_input(inputs):
            user_message = inputs["user_message"]
            chat_history = inputs.get("chat_history", [])
            context = inputs.get("context", {})
            
            # Combine into the format expected by the agent executor
            return {
                "input": user_message,
                "chat_history": chat_history,
                "context": context
            }
        
        # Create output formatting function
        def format_output(output):
            if "output" in output:
                parsed_output = self._parse_agent_output(output["output"])
                return {
                    "response": parsed_output,
                    "raw_output": output
                }
            return {
                "response": self._parse_agent_output(str(output)),
                "raw_output": output
            }
        
        # Create the LCEL chain
        chain = (
            RunnablePassthrough()
            .assign(prepared_input=RunnableLambda(prepare_input))
            .assign(agent_result=lambda x: agent_executor.invoke(x["prepared_input"]))
            .assign(formatted_output=lambda x: format_output(x["agent_result"]))
            .pick(["formatted_output"])
        )
        
        return chain
    
    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history from the database.
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            List of message objects
        """
        # Get recent messages (limit to a reasonable number to avoid context length issues)
        messages = self.db_client.get_conversation_messages(conversation_id, limit=10)
        
        # Convert to LangChain message format
        history = []
        for msg in messages:
            if msg["sender"] == "user":
                if msg["kind"] == "text":
                    body = json.loads(msg["body"]) if isinstance(msg["body"], str) and msg["body"].startswith("{") else msg["body"]
                    content = body["text"] if isinstance(body, dict) and "text" in body else msg["body"]
                    history.append(HumanMessage(content=content))
            elif msg["sender"] == "assistant":
                if msg["kind"] == "text":
                    body = json.loads(msg["body"]) if isinstance(msg["body"], str) and msg["body"].startswith("{") else msg["body"]
                    content = body["text"] if isinstance(body, dict) and "text" in body else msg["body"]
                    history.append(AIMessage(content=content))
                    
        return history
    
    async def process_user_message(self, conversation_id: str, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            conversation_id: The ID of the conversation
            user_message: The text message from the user
            context: Additional context like user_id
            
        Returns:
            Response from the assistant
        """
        # Get conversation history
        history = await self._get_conversation_history(conversation_id)
        
        # Run the agent chain
        result = await self.chain.ainvoke({
            "user_message": user_message,
            "chat_history": history,
            "context": context
        })
        
        # Extract the formatted output
        output = result["formatted_output"]["response"]
        
        # Store the assistant's response in the database
        assistant_message = self.db_client.create_message(
            conversation_id=str(conversation_id),
            sender="assistant",
            kind=output.get("kind", "text"),
            body=json.dumps(output)
        )
        
        return {
            "message": assistant_message,
            "response": output
        }
    
    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        """
        Parse the agent's output to ensure it's in the correct format.
        If it's not valid JSON, convert it to a text response.
        
        Args:
            output: The output string from the agent
            
        Returns:
            Structured output dict
        """
        # Try to parse as JSON
        try:
            # Check if output is already a dictionary
            if isinstance(output, dict):
                # Validate it has the required fields
                if "kind" in output:
                    return output
                else:
                    # Convert to text response
                    return {"kind": "text", "text": str(output)}
            
            # Extract JSON if it's inside a code block
            if "```json" in output:
                json_text = output.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_text)
                
                # Validate it has the required fields
                if "kind" in parsed:
                    return parsed
            
            # Try to parse the entire output as JSON
            parsed = json.loads(output)
            
            # Validate it has the required fields
            if "kind" in parsed:
                return parsed
                
        except (json.JSONDecodeError, IndexError, KeyError):
            # If parsing fails or required fields are missing, convert to text
            pass
        
        # Default to text response
        return {"kind": "text", "text": output}
    
    async def process_action(self, conversation_id: str, action: str, values: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an action from the user (e.g., button click or form submission).
        
        Args:
            conversation_id: The ID of the conversation
            action: The action identifier
            values: Values associated with the action
            context: Additional context like user_id
            
        Returns:
            Response from the assistant
        """
        # Store the action in the database
        action_message = self.db_client.create_message(
            conversation_id=str(conversation_id),
            sender="user",
            kind="action",
            body=json.dumps({
                "action": action,
                "values": values
            })
        )
        
        # Format the action as input for the agent
        action_input = f"User performed action: {action}"
        if values:
            action_input += f" with values: {json.dumps(values)}"
        
        # Process the action with the agent
        response = await self.process_user_message(
            conversation_id=conversation_id,
            user_message=action_input,
            context=context
        )
        
        return response
    
    async def process_file_upload(self, conversation_id: str, file_id: str, file_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a file upload from the user.
        
        Args:
            conversation_id: The ID of the conversation
            file_id: The ID of the uploaded file
            file_info: Information about the file
            context: Additional context like user_id
            
        Returns:
            Response from the assistant
        """
        # Store the file message in the database
        file_message = self.db_client.create_message(
            conversation_id=str(conversation_id),
            sender="user",
            kind="file",
            body=json.dumps({
                "file_id": file_id,
                "name": file_info.get("name", ""),
                "mime": file_info.get("mime", ""),
                "size": file_info.get("size", 0),
                "bucket": file_info.get("bucket", "user-uploads"),
                "object_key": file_info.get("object_key", "")
            })
        )
        
        # Format the file upload as input for the agent
        file_input = f"User uploaded a file: {file_info.get('name', '')} (type: {file_info.get('mime', '')})"
        
        # Process the file upload with the agent
        response = await self.process_user_message(
            conversation_id=conversation_id,
            user_message=file_input,
            context={**context, "file_id": file_id}
        )
        
        return response