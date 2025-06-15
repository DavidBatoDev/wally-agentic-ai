# backend/src/agent/functions/analyze_doc_helper.py
"""
Document analysis client using Gemini API for content extraction and language detection.

This module contains exactly one public coroutine, `analyze_upload`, which
downloads the file from the public URL, sends it to Gemini API for analysis,
and returns structured analysis as a Python dict.

* Uses Gemini 2.0 Flash model for document analysis
* Supports images, PDFs, and Word documents
* Automatic language detection and content extraction
* Environment variable `GEMINI_API_KEY` required for authentication
"""

from __future__ import annotations

import os
import logging
import base64
import mimetypes
from typing import Any, Dict
from urllib.parse import urlparse
import json

import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT", "60"))
RETRIES = int(os.getenv("GEMINI_RETRIES", "3"))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _detect_mime_type(public_url: str) -> str:
    """Detect MIME type from URL extension."""
    parsed = urlparse(public_url)
    path = parsed.path.lower()
    
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        return mime_type
    
    # Fallback for common extensions
    if path.endswith('.pdf'):
        return 'application/pdf'
    elif path.endswith('.docx'):
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif path.endswith('.doc'):
        return 'application/msword'
    elif path.endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    elif path.endswith('.png'):
        return 'image/png'
    elif path.endswith('.webp'):
        return 'image/webp'
    else:
        return 'application/octet-stream'

def _determine_doc_type(mime_type: str) -> str:
    """Determine document type based on MIME type."""
    if mime_type.startswith('image/'):
        return 'image'
    elif mime_type == 'application/pdf':
        return 'pdf'
    elif 'word' in mime_type or mime_type == 'application/msword':
        return 'document'
    elif 'sheet' in mime_type:
        return 'spreadsheet'
    elif mime_type == 'text/plain':
        return 'text'
    else:
        return 'unknown'


async def _download_file(url: str) -> bytes:
    """Download file from URL and return as bytes."""
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _create_gemini_payload(file_data: bytes, mime_type: str) -> Dict[str, Any]:
    """Create the payload for Gemini API request."""
    # Encode file data as base64
    file_b64 = base64.b64encode(file_data).decode('utf-8')
    
    # Create the enhanced analysis prompt with specific focus on PSA/birth certificates
    prompt = """
Analyze this document thoroughly and provide a JSON response with the following information:
- detected_language: ISO 639-1 language code (e.g., "en", "es", "fr", "de", "zh", etc.)
- confidence: float between 0.0 and 1.0 indicating confidence in the analysis
- page_count: exact number of pages (integer) - be precise about this count
- page_size: estimated page size (e.g., "A4", "Letter", "Legal", "variable")
- content_summary: detailed explanation of the document content and structure (see below for requirements)
- doc_classification: classification like "psa", "birth_certificate", "death_certificate", "marriage_certificate", "id", "passport", "driver_license", "visa", "certificate", "diploma", "transcript", "contract", "agreement", "invoice", "receipt", "report", "letter", "form", "application", "prescription", "medical_document", "legal_document", "business_document", "academic_document", "official_document", "other"
- is_psa_document: boolean indicating if this appears to be a PSA (Philippine Statistics Authority) document
- has_psa_features: boolean indicating if document has PSA-specific features (security paper, official seals, QR codes, reference numbers, etc.)
- is_templatable: boolean indicating if this PSA document contains actual personal data/values that can be extracted for templating (not blank forms)

For the content_summary field, provide a comprehensive explanation that includes:
1. What type of document this is and its primary purpose
2. Key sections or areas identified in the document
3. Important fields, data, or information present
4. The document's structure and layout
5. Any notable features, stamps, seals, or formatting
6. Context about how this document might be used
7. Any specific requirements or standards this document appears to follow

Make the content_summary detailed and informative (aim for 300-500 characters) as it will be shown to users to help them understand what was detected in their document.

SPECIAL FOCUS: Pay close attention to identifying PSA documents and determining if they are templatable:
- Look for "PSA" text, logos, or watermarks
- Check for security features like special paper texture, watermarks
- Look for official seals, stamps, or government insignia
- Check for reference numbers, QR codes, or barcodes
- Identify birth certificate specific fields (name, birth date, place of birth, parents' names, etc.)
- Note any Philippine government formatting or standards

CRITICAL FOR is_templatable DETERMINATION:
- Set is_templatable to TRUE only if:
  1. The document is clearly a PSA document (birth certificate, death certificate, marriage certificate, etc.)
  2. AND the document contains actual filled-in personal information/values with real names, dates, places
  3. AND you can see COMPLETED fields with actual data (not placeholders, templates, or variables like {name}, {date}, etc.)
  4. AND the document appears to be 1-2 pages maximum
  5. AND your confidence level is at least 0.6

- Set is_templatable to FALSE if:
  1. It's not a PSA document
  2. OR it's a blank/empty form without personal data
  3. OR it contains placeholder text, template variables (like {name}, {province}, etc.), or fill-in-the-blank formatting
  4. OR it's a form template rather than a completed document
  5. OR it's more than 2 pages
  6. OR confidence is below 0.6
  7. OR you cannot clearly identify actual completed personal information

IMPORTANT: Pay special attention to distinguish between:
- BLANK FORMS/TEMPLATES: Contains placeholders like {name}, {date}, empty fields, or template formatting → NOT templatable
- COMPLETED DOCUMENTS: Contains actual names, real dates, filled-in information → Potentially templatable

For page_count: Be very precise. Count actual document pages, not including blank pages or covers.

IMPORTANT: Be extremely accurate about the page count, document classification, and especially the is_templatable determination. The system relies on your assessment to determine processing workflow.

Return only valid JSON without any markdown formatting or additional text.
"""
    
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": file_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2000
        }
    }


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(RETRIES),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _call_gemini_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Gemini API with retry logic."""
    headers = {
        "Content-Type": "application/json"
    }
    
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the generated text from Gemini response
        if "candidates" in result and len(result["candidates"]) > 0:
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse the JSON response from Gemini
            try:
                # Clean up the response - remove any markdown formatting
                clean_text = generated_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                return json.loads(clean_text)
            except json.JSONDecodeError as e:
                log.error("Failed to parse Gemini JSON response: %s", generated_text)
                raise ValueError(f"Invalid JSON response from Gemini: {e}")
        else:
            raise ValueError("No content generated by Gemini API")


# ---------------------------------------------------------------------------
# Public coroutine
# ---------------------------------------------------------------------------

def _normalize_url(maybe_url: str) -> str:
    parsed = urlparse(maybe_url)
    # If there's no scheme, assume HTTPS
    if not parsed.scheme:
        return "https://" + maybe_url
    return maybe_url

async def analyze_upload(file_id: str, public_url: str) -> Dict[str, Any]:
    """Download file and analyze it using Gemini API."""

    try:
        # Detect MIME type and document type
        public_url = _normalize_url(public_url)
        mime_type = _detect_mime_type(public_url)
        doc_type = _determine_doc_type(mime_type)
        
        log.debug("Detected → mime_type: %s, doc_type: %s", mime_type, doc_type)
        
        # Download the file
        log.debug("Downloading file from: %s", public_url)
        file_data = await _download_file(public_url)
        
        # Create Gemini API payload
        payload = _create_gemini_payload(file_data, mime_type)
        
        # Call Gemini API for analysis
        log.debug("Calling Gemini API for analysis")
        gemini_result = await _call_gemini_api(payload)
        
        # Ensure content_summary has a fallback if not provided or too short
        content_summary = gemini_result.get("content_summary", "")
        doc_classification = gemini_result.get("doc_classification", "other")
        page_count = gemini_result.get("page_count", 0)
        confidence = float(gemini_result.get("confidence", 0.0))
        
        if not content_summary or len(content_summary.strip()) < 50:
            # Generate a basic fallback summary based on doc classification and type
            detected_language = gemini_result.get("detected_language", "unknown")
            
            content_summary = (
                f"This appears to be a {doc_classification} in {detected_language} language. "
                f"The document has been processed and analyzed for content extraction and translation purposes. "
                f"Key information and data fields have been identified for further processing."
            )
        
        # Let Gemini decide if the document is templatable based on PSA content with personal values
        is_templatable = gemini_result.get("is_templatable", False)
        
        log.debug("Gemini determined is_templatable: %s for doc_classification: %s", is_templatable, doc_classification)
        
        # Build the final analysis result
        analysis_result = {
            "file_id": file_id,
            "mime_type": mime_type,
            "doc_type": doc_type,
            "variation": "standard",
            "detected_language": gemini_result.get("detected_language", "unknown"),
            "confidence": confidence,
            "page_count": page_count,
            "page_size": gemini_result.get("page_size"),
            "analysis_timestamp": None,
            "content_summary": content_summary,
            "doc_classification": doc_classification,
            "is_templatable": is_templatable,
            "is_psa_document": gemini_result.get("is_psa_document", False),
            "has_psa_features": gemini_result.get("has_psa_features", False)
        }
        
        log.debug("Analysis complete → %s", {k: v for k, v in analysis_result.items() if k != "content_summary"})
        log.debug("Content summary length: %d characters", len(content_summary))
        log.debug("Document is_templatable: %s", is_templatable)
        return analysis_result
        
    except Exception as e:
        log.error("Analysis failed for file_id %s: %s", file_id, str(e))
        
        # Return fallback analysis on error with detailed explanation
        mime_type = _detect_mime_type(public_url)
        doc_type = _determine_doc_type(mime_type)
        
        fallback_summary = (
            f"Document analysis encountered technical difficulties, but the file appears to be a {doc_type} "
            f"based on its format. The system will proceed with standard processing methods. "
            f"Manual verification of extracted content may be recommended."
        )
        
        # Conservative fallback - assume not templatable unless we can verify it meets criteria
        fallback_is_templatable = False
        
        return {
            "file_id": file_id,
            "mime_type": mime_type,
            "doc_type": doc_type,
            "variation": "standard",
            "detected_language": "unknown",
            "confidence": 0.0,
            "page_count": None,
            "page_size": None,
            "analysis_timestamp": None,
            "content_summary": fallback_summary,
            "doc_classification": "other",
            "is_templatable": fallback_is_templatable,
            "is_psa_document": False,
            "has_psa_features": False
        }