# backend/src/agent/functions/find_template_node_helper.py
"""
Template matching client using Gemini API for finding the best template match.

This module contains exactly one public coroutine, `find_template_match`, which
fetches available templates from the database, cleans the data, and uses Gemini API
to determine the best matching template based on the uploaded document analysis.

* Uses Gemini 2.0 Flash model for template matching
* Analyzes document against available templates
* Returns template_id, template_required_fields, and process analysis
* Environment variable `GEMINI_API_KEY` required for authentication
"""

from __future__ import annotations

import os
import logging
import json
from typing import Any, Dict, List, Optional
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

def _clean_template_data(templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean template data by removing large fields like required_fields and fillable_text_info."""
    cleaned_templates = []
    
    for template in templates:
        cleaned_template = {
            "id": template.get("id"),
            "file_url": template.get("file_url"),
            "doc_type": template.get("doc_type"),
            "variation": template.get("variation"),
            "created_at": template.get("created_at")
        }
        
        # Clean info_json by removing large fields
        if "info_json" in template and template["info_json"]:
            info_json = template["info_json"].copy()
            
            # Remove the large fields
            info_json.pop("required_fields", None)
            info_json.pop("fillable_text_info", None)
            
            cleaned_template["info_json"] = info_json
        
        cleaned_templates.append(cleaned_template)
    
    return cleaned_templates


def _create_gemini_payload(
    upload_analysis: Dict[str, Any],
    cleaned_templates: List[Dict[str, Any]],
    translated_to: Optional[str] = None
) -> Dict[str, Any]:
    """Create the payload for Gemini API request."""
    
    # Create the template matching prompt
    prompt = f"""
You are a document template matching expert. Analyze the uploaded document information and find the best matching template from the available options.

UPLOADED DOCUMENT ANALYSIS:
{json.dumps(upload_analysis, indent=2)}

AVAILABLE TEMPLATES:
{json.dumps(cleaned_templates, indent=2)}

TARGET TRANSLATION LANGUAGE: {translated_to or "Not specified"}

Your task is to:
1. Compare the uploaded document characteristics with available templates
2. Consider document type, variation, language, and content classification
3. **IMPORTANT**: If a translation language is specified, give HIGHER PRIORITY to templates where info_json.language matches the translated_to language exactly
4. Language matching should significantly boost confidence scores (add 0.3-0.4 to base confidence)
5. Select the most appropriate template match
6. Provide detailed reasoning for your selection

Return a JSON response with the following structure:
{{
    "template_match_found": boolean,
    "template_id": "template_id_if_found_or_null",
    "confidence_score": float_between_0_and_1,
    "match_reasoning": "detailed_explanation_of_why_this_template_was_selected_including_language_match_bonus",
    "language_match_bonus": boolean_indicating_if_template_language_matches_translated_to,
    "process_analysis": {{
        "document_compatibility": "assessment_of_how_well_document_matches_template",
        "language_alignment": "analysis_of_language_compatibility_and_translation_target_match",
        "structural_similarity": "evaluation_of_document_structure_match",
        "recommended_next_steps": "suggestions_for_processing_this_document"
    }}
}}

Matching criteria priority (in order):
1. **Language Match**: Template info_json.language exactly matches translated_to (MAJOR confidence boost)
2. Document type and classification match
3. Document variation/format similarity  
4. Content structure and purpose alignment
5. General language family compatibility

CONFIDENCE SCORING GUIDELINES:
- Base match (type + structure): 0.3-0.6
- Language exact match bonus: +0.3-0.4 
- Strong structural similarity: +0.1-0.2
- Perfect match (all criteria): 0.8-0.95

Examples:
- If translated_to="English" and template has info_json.language="English" → high confidence boost
- If translated_to="Spanish" and template has info_json.language="Spanish" → high confidence boost
- If no language match but good type/structure match → moderate confidence

If no suitable template is found, set template_match_found to false and template_id to null, but still provide analysis.

Return only valid JSON without any markdown formatting or additional text.
"""
    
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
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


async def _fetch_all_templates(db_client) -> List[Dict[str, Any]]:
    """Fetch all templates from the templates table."""
    try:
        response = db_client.client.table("templates").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        log.error("Failed to fetch templates from database: %s", str(e))
        return []


async def _fetch_template_by_id(db_client, template_id: str) -> Dict[str, Any]:
    """Fetch a specific template by ID."""
    try:
        response = db_client.client.table("templates").select("*").eq("id", template_id).execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        log.error("Failed to fetch template by ID %s: %s", template_id, str(e))
        return {}


async def _fetch_template_required_fields(db_client, template_id: str) -> Dict[str, str]:
    """Fetch the required_fields for a specific template."""
    try:
        template = await _fetch_template_by_id(db_client, template_id)
        
        if template and "info_json" in template and template["info_json"]:
            return template["info_json"].get("required_fields", {})
        
        return {}
    except Exception as e:
        log.error("Failed to fetch template required fields for %s: %s", template_id, str(e))
        return {}


# ---------------------------------------------------------------------------
# Public coroutine
# ---------------------------------------------------------------------------

async def find_template_match(
    db_client,
    upload_analysis: Dict[str, Any],
    user_upload_public_url: str,
    translated_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find the best matching template for the uploaded document.
    
    Args:
        db_client: Database client instance
        upload_analysis: Analysis result from analyze_upload
        user_upload_public_url: URL of the uploaded file
        translated_to: Target translation language (optional)
    
    Returns:
        Dict containing template_id, template_required_fields, and process_analysis
    """
    
    try:
        # Step 1: Fetch all templates from database
        log.debug("Fetching all templates from database")
        templates = await _fetch_all_templates(db_client)
        
        if not templates:
            log.warning("No templates found in database")
            return {
                "template_match_found": False,
                "template_id": None,
                "template_required_fields": {},
                "confidence_score": 0.0,
                "language_match_bonus": False,
                "match_reasoning": "No templates available for matching",
                "process_analysis": {
                    "document_compatibility": "No templates available for matching",
                    "language_alignment": "Cannot assess without templates",
                    "structural_similarity": "No templates to compare against",
                    "recommended_next_steps": "Add templates to the database for document processing"
                }
            }
        
        # Step 2: Clean template data
        log.debug("Cleaning template data, found %d templates", len(templates))
        cleaned_templates = _clean_template_data(templates)
        
        # Step 3: Create Gemini API payload
        payload = _create_gemini_payload(upload_analysis, cleaned_templates, translated_to)
        
        # Step 4: Call Gemini API for template matching
        log.debug("Calling Gemini API for template matching")
        gemini_result = await _call_gemini_api(payload)
        
        # Step 5: If a template was found, fetch its required fields
        template_required_fields = {}
        template_id = gemini_result.get("template_id")
        
        if gemini_result.get("template_match_found") and template_id:
            log.debug("Template match found: %s", template_id)
            template_required_fields = await _fetch_template_required_fields(db_client, template_id)
        
        # Step 6: Build the final result
        result = {
            "template_match_found": gemini_result.get("template_match_found", False),
            "template_id": template_id,
            "template_required_fields": template_required_fields,
            "confidence_score": float(gemini_result.get("confidence_score", 0.0)),
            "language_match_bonus": gemini_result.get("language_match_bonus", False),
            "match_reasoning": gemini_result.get("match_reasoning", "No reasoning provided"),
            "process_analysis": gemini_result.get("process_analysis", {
                "document_compatibility": "Analysis incomplete",
                "language_alignment": "Analysis incomplete", 
                "structural_similarity": "Analysis incomplete",
                "recommended_next_steps": "Retry template matching process"
            })
        }
        
        log.debug("Template matching complete → match_found: %s, template_id: %s, confidence: %.2f, language_bonus: %s",
                 result["template_match_found"], result["template_id"], result["confidence_score"], result["language_match_bonus"])
        
        return result
        
    except Exception as e:
        log.error("Template matching failed: %s", str(e))
        
        # Return fallback result on error
        return {
            "template_match_found": False,
            "template_id": None,
            "template_required_fields": {},
            "confidence_score": 0.0,
            "language_match_bonus": False,
            "match_reasoning": f"Template matching failed due to technical error: {str(e)}",
            "process_analysis": {
                "document_compatibility": f"Template matching failed due to technical error: {str(e)}",
                "language_alignment": "Could not assess due to processing error",
                "structural_similarity": "Analysis interrupted by system error",
                "recommended_next_steps": "Retry the template matching process or contact support if the issue persists"
            }
        }