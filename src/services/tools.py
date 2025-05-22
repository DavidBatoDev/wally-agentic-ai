import os
import json
import httpx
import base64
from typing import Dict, List, Union, Optional, Any
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

class ExtractDocumentInput(BaseModel):
    """Input for extracting information from a document."""
    file_path: str = Field(..., description="Local path to the document file (PDF, DOCX, or image)")
    template_id: Optional[str] = Field(None, description="Optional template ID to use for extraction")

class FillTemplateInput(BaseModel):
    """Input for filling a template with values."""
    template_id: str = Field(..., description="UUID of template row in Supabase")
    values: Dict[str, str] = Field(..., description="Mapping of placeholders to values (with {{ }} in keys)")

class DocumentExtractorTool(BaseTool):
    """Tool for extracting information from documents using OCR."""
    name = "document_extractor"
    description = """
    Extract information from documents (PDF, DOCX, or images) using OCR.
    Use this tool when you need to extract structured information from a document file.
    The tool uses Gemini API to perform OCR and extract information according to a template.
    Input should be a local file path and optionally a template ID.
    """
    args_schema = ExtractDocumentInput
    api_url: str = "http://localhost:8000/api/extract"  # Default URL
    
    def __init__(self, api_url: Optional[str] = None):
        """Initialize with optional custom API URL."""
        super().__init__()
        if api_url:
            self.api_url = api_url
    
    def _run(
        self, 
        file_path: str, 
        template_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Run the document extraction tool."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Prepare the file for upload
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Prepare the form data
        files = {"file": (os.path.basename(file_path), file_content)}
        data = {}
        if template_id:
            data["template_id"] = template_id
        
        # Make the API request
        try:
            response = httpx.post(
                self.api_url, 
                files=files,
                data=data,
                timeout=30.0  # Longer timeout for OCR processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "message": response.text
                }
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}
    
    async def _arun(
        self, 
        file_path: str, 
        template_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Run the document extraction tool asynchronously."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Prepare the file for upload
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Prepare the form data
        files = {"file": (os.path.basename(file_path), file_content)}
        data = {}
        if template_id:
            data["template_id"] = template_id
        
        # Make the API request asynchronously
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url, 
                    files=files,
                    data=data,
                    timeout=30.0  # Longer timeout for OCR processing
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": f"API request failed with status {response.status_code}",
                        "message": response.text
                    }
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}

class TemplateFillerTool(BaseTool):
    """Tool for filling templates with values."""
    name = "template_filler"
    description = """
    Fill a template document with provided values.
    Use this tool when you need to generate a filled document from a template.
    Input should include the template ID and a mapping of placeholders to values.
    Returns a path to the downloaded filled document.
    """
    args_schema = FillTemplateInput
    api_url: str = "http://localhost:8000/api/fill"  # Default URL
    
    def __init__(self, api_url: Optional[str] = None, download_dir: str = "./downloads"):
        """Initialize with optional custom API URL and download directory."""
        super().__init__()
        if api_url:
            self.api_url = api_url
        self.download_dir = download_dir
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
    
    def _run(
        self, 
        template_id: str, 
        values: Dict[str, str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Run the template filler tool."""
        # Format values for the API - ensure all keys have {{ }}
        formatted_values = {}
        for key, value in values.items():
            # Add {{ }} if not already present
            if not key.startswith("{{"):
                key = "{{" + key
            if not key.endswith("}}"):
                key = key + "}}"
            formatted_values[key] = value
        
        # Convert values to JSON string
        payload = json.dumps(formatted_values)
        
        # Make the API request
        try:
            response = httpx.post(
                self.api_url,
                params={"id": template_id, "payload": payload},
                timeout=20.0
            )
            
            if response.status_code == 200:
                # Save the received document to a file
                output_path = os.path.join(self.download_dir, f"filled_{template_id}.docx")
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return {
                    "success": True,
                    "message": f"Document filled successfully",
                    "file_path": output_path
                }
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "message": response.text
                }
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}
    
    async def _arun(
        self, 
        template_id: str, 
        values: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Run the template filler tool asynchronously."""
        # Format values for the API - ensure all keys have {{ }}
        formatted_values = {}
        for key, value in values.items():
            # Add {{ }} if not already present
            if not key.startswith("{{"):
                key = "{{" + key
            if not key.endswith("}}"):
                key = key + "}}"
            formatted_values[key] = value
        
        # Convert values to JSON string
        payload = json.dumps(formatted_values)
        
        # Make the API request asynchronously
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    params={"id": template_id, "payload": payload},
                    timeout=20.0
                )
                
                if response.status_code == 200:
                    # Save the received document to a file
                    output_path = os.path.join(self.download_dir, f"filled_{template_id}.docx")
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    return {
                        "success": True,
                        "message": f"Document filled successfully",
                        "file_path": output_path
                    }
                else:
                    return {
                        "error": f"API request failed with status {response.status_code}",
                        "message": response.text
                    }
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}

def get_document_processing_tools(
    api_base_url: Optional[str] = None,
    download_dir: str = "./downloads"
) -> List[BaseTool]:
    """
    Get a list of document processing tools configured with the given API base URL.
    
    Args:
        api_base_url: Base URL for the document processing API
                     (default: http://localhost:8000)
        download_dir: Directory to save filled documents
                     (default: ./downloads)
    
    Returns:
        List of document processing tools
    """
    if api_base_url is None:
        api_base_url = "http://localhost:8000"
    
    extract_url = f"{api_base_url}/api/extract"
    fill_url = f"{api_base_url}/api/fill"
    
    return [
        DocumentExtractorTool(api_url=extract_url),
        TemplateFillerTool(api_url=fill_url, download_dir=download_dir)
    ]