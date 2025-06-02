# backend/src/agent/analyze_tools.py
import json
import logging
from typing import Dict, Any
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import fitz   # PyMuPDF
import requests
import tempfile
import os
import mimetypes

log = logging.getLogger(__name__)

class AnalyzeUploadInput(BaseModel):
    file_id: str = Field(..., description="Supabase UUID of the uploaded file")
    public_url: str = Field(..., description="Signed or public URL to download the file")
    # llm: Any = Field(..., description="LLM instance for analysis")

def analyze_upload_with_llm(file_id: str, public_url: str, llm: Any) -> Dict:
    """
    Download file, extract basic metadata, then use LLM to analyze and determine document type.
    Returns structured analysis data for the agent state.
    """
    try:
        # Download and analyze file
        tmp_file = tempfile.mktemp()
        
        # Download file
        response = requests.get(public_url)
        response.raise_for_status()
        
        with open(tmp_file, "wb") as f:
            f.write(response.content)

        # Extract basic metadata
        mime_type = mimetypes.guess_type(tmp_file)[0] or "application/pdf"
        file_size = os.path.getsize(tmp_file)
        
        # For PDF files, extract additional metadata
        page_info = {}
        text_content = ""
        
        if mime_type == "application/pdf":
            try:
                doc = fitz.open(tmp_file)
                page_count = len(doc)
                
                # Get first page dimensions
                if page_count > 0:
                    first_page = doc[0]
                    width, height = first_page.rect.width, first_page.rect.height
                    page_info = {
                        "page_count": page_count,
                        "page_size": f"{width:.0f}×{height:.0f}",
                        "orientation": "landscape" if width > height else "portrait"
                    }
                    
                    # Extract some text for analysis (first 1000 chars)
                    text_content = first_page.get_text()[:1000]
                
                doc.close()
            except Exception as e:
                log.warning(f"Error extracting PDF metadata: {e}")
        
        # Clean up temp file
        os.remove(tmp_file)
        
        # Prepare context for LLM analysis
        analysis_context = {
            "file_id": file_id,
            "mime_type": mime_type,
            "file_size_bytes": file_size,
            "text_sample": text_content,
            **page_info
        }
        
        # Use LLM to analyze the document
        system_prompt = """You are a document analysis expert. Based on the provided file metadata and text content, determine:

1. Document type (e.g., "passport", "national_id", "driver_license", "birth_certificate", "marriage_certificate", "diploma", "transcript", "medical_record", "bank_statement", "utility_bill", "employment_letter", "other")
2. Document variation/subtype (e.g., "standard", "enhanced", "old_format", "new_format", "horizontal", "vertical")
3. Detected language (e.g., "english", "spanish", "french", "tagalog", "cebuano", "unknown")
4. Confidence level (0.0 to 1.0)

Respond with a JSON object containing these fields:
{
    "doc_type": "document_type_here",
    "variation": "variation_here", 
    "detected_language": "language_here",
    "confidence": 0.85
}

Be conservative with confidence scores. Use "unknown" when uncertain."""

        user_prompt = f"""Analyze this document:

File metadata:
- MIME type: {analysis_context['mime_type']}
- File size: {analysis_context['file_size_bytes']:,} bytes
- Page count: {analysis_context.get('page_count', 'unknown')}
- Page size: {analysis_context.get('page_size', 'unknown')}
- Orientation: {analysis_context.get('orientation', 'unknown')}

Text content sample:
{analysis_context['text_sample'][:500]}...

Provide your analysis as JSON."""

        # Get LLM analysis
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        llm_response = llm.invoke(messages)
        
        # Parse LLM response
        try:
            # Try to extract JSON from the response
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                llm_analysis = json.loads(json_match.group())
            else:
                # Fallback parsing
                llm_analysis = json.loads(response_text)
                
        except (json.JSONDecodeError, AttributeError) as e:
            log.warning(f"Could not parse LLM response as JSON: {e}")
            # Fallback to basic analysis
            llm_analysis = {
                "doc_type": "unknown",
                "variation": "standard",
                "detected_language": "unknown",
                "confidence": 0.1
            }
        
        # Combine metadata with LLM analysis
        final_analysis = {
            "file_id": file_id,
            "mime_type": mime_type,
            "doc_type": llm_analysis.get("doc_type", "unknown"),
            "variation": llm_analysis.get("variation", "standard"),
            "detected_language": llm_analysis.get("detected_language", "unknown"),
            "confidence": llm_analysis.get("confidence", 0.5),
            "page_size": analysis_context.get("page_size", "unknown"),
            "page_count": analysis_context.get("page_count", 1),
            "file_size_bytes": file_size,
            "analysis_timestamp": None,  # Will be set by the system
        }
        
        return final_analysis
        
    except Exception as e:
        log.exception(f"Error analyzing upload {file_id}: {e}")
        # Return minimal analysis on error
        return {
            "file_id": file_id,
            "mime_type": "unknown",
            "doc_type": "unknown", 
            "variation": "standard",
            "detected_language": "unknown",
            "confidence": 0.0,
            "error": str(e)
        }

def get_analyze_tool(llm) -> StructuredTool:
    """
    Create the analyze_upload tool with the LLM instance.
    This tool will be used by the agent to analyze uploaded documents.
    """
    def analyze_wrapper(file_id: str, public_url: str) -> Dict[str, Any]:
        return analyze_upload_with_llm(file_id, public_url, llm)
    
    return StructuredTool.from_function(
        func=analyze_wrapper,
        name="analyze_upload",
        description="Analyze an uploaded document to determine its type, variation, and language using AI",
        args_schema=AnalyzeUploadInput,
        return_direct=True,  # Return results directly to the agent
    )