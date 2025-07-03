# backend/src/agent/functions/find_template_node_helper.py
"""
Template matching helper for finding original and translated template pairs.

This module provides the `find_template_match` function that:
1. Analyzes the uploaded document's characteristics using LLM
2. Finds the appropriate original template based on document layout comparison
3. Finds the corresponding translated template for the target language
4. Returns structured template information for workflow processing

The function uses LLM to compare document layouts and match templates based on:
- Document type (e.g., birth_certificate)
- Language variations (source and target languages)
- Year variations and layout compatibility
- Template compatibility with the analyzed document
"""

from __future__ import annotations

import logging
import os
import requests
import base64
import io
from typing import Dict, Any, Tuple, Optional, List
from pdf2image import convert_from_bytes
from PIL import Image
from ..agent_state import AgentState, Upload

log = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyAsIInmf09IT7YgVRHgEmS19dT4uIYdJkQ"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT", "60"))
RETRIES = int(os.getenv("GEMINI_RETRIES", "3"))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Language mapping for template matching
LANGUAGE_TO_TEMPLATE_VARIATION = {
    "en": "English",
    "english": "English", 
    "el": "Greek",
    "greek": "Greek",
    "gr": "Greek",
    "es": "Spanish",
    "spanish": "Spanish",
    "fr": "French", 
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "zh": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese", 
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "ar": "Arabic",
    "arabic": "Arabic",
    "ru": "Russian",
    "russian": "Russian",
    "hi": "Hindi",
    "hindi": "Hindi",
    "th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "ms": "Malay",
    "malay": "Malay",
    "tl": "Filipino",
    "filipino": "Filipino",
    "fil": "Filipino"
}

def _normalize_language_for_template(language: str) -> str:
    """Normalize language code/name to template variation format."""
    if not language:
        return "English"  # Default fallback
    
    normalized = language.lower().strip()
    return LANGUAGE_TO_TEMPLATE_VARIATION.get(normalized, "English")

def _determine_doc_type_for_template(upload: Upload) -> str:
    """Determine the document type for template matching based on analysis."""
    if not upload.analysis:
        return "birth_certificate"  # Default fallback
    
    doc_classification = upload.analysis.get("doc_classification", "")
    
    # Map analysis classifications to template doc_types
    classification_mapping = {
        "psa": "birth_certificate",
        "birth_certificate": "birth_certificate", 
        "death_certificate": "death_certificate",
        "marriage_certificate": "marriage_certificate",
        "certificate": "birth_certificate",  # Default to birth cert for generic certificates
        "official_document": "birth_certificate"
    }
    
    return classification_mapping.get(doc_classification, "birth_certificate")

def _load_sample_templates() -> List[Dict[str, Any]]:
    """Load the sample templates from hardcoded JSON data."""
    try:
        templates = [
            {
                "id": "d60347cb-0131-4028-aa22-4fbcb1c2775e",
                "doc_type": "birth_certificate",
                "variation": "Greek_1993_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201993%20Greek.pdf"
            },
            {
                "id": "720be31c-6a57-4968-83ef-955f9f44caf7",
                "doc_type": "birth_certificate",
                "variation": "Romanian_1993_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201993%20Romanian.pdf"
            },
            {
                "id": "3174e9a6-182c-4619-a022-db3afddfb112",
                "doc_type": "birth_certificate",
                "variation": "Japanese_1993_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201993%20Japanese.pdf"
            },
            {
                "id": "a9b6293b-6b7c-49f0-adca-57efd105b387",
                "doc_type": "birth_certificate",
                "variation": "Korean_1993_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201993%20Korean.pdf"
            },
            {
                "id": "9fc0c5fc-2885-4d58-ba0f-4711244eb7df",
                "doc_type": "birth_certificate",
                "variation": "English_1993_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201993%20ENG%20blank%20w%202nd%20page.pdf"
            },
            {
                "id": "f40727f1-4f56-4892-8548-9e77be594775",
                "doc_type": "birth_certificate",
                "variation": "English_1958_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201958%20English.pdf"
            },
            {
                "id": "ef24c1f5-a573-4541-b44a-c81531275d05",
                "doc_type": "birth_certificate",
                "variation": "Greek_1958_template",
                "file_url": "https://ylvmwrvyiamecvnydwvj.supabase.co/storage/v1/object/public/templates/templates/PSA%20Birth%20Cert%201958%20Greek.pdf"
            }
        ]
        
        log.debug(f"Loaded {len(templates)} sample templates")
        return templates
    except Exception as e:
        log.error(f"Error loading sample templates: {str(e)}")
        return []

def _filter_templates_by_language(templates: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    """Filter templates based on language variation."""
    if not language or not templates:
        return []
    
    # Normalize the language
    normalized_language = _normalize_language_for_template(language)
    
    # Filter templates where variation starts with the normalized language
    filtered_templates = []
    for template in templates:
        variation = template.get("variation", "")
        if variation:
            variation_language = variation.split("_")[0]
            if variation_language.lower() == normalized_language.lower():
                filtered_templates.append(template)
    
    log.debug(f"Filtered {len(filtered_templates)} templates for language: {normalized_language}")
    return filtered_templates

async def _llm_compare_layouts(
    user_upload_url: str,
    template_urls: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to compare user upload layout with template layouts using image analysis.
    
    Args:
        user_upload_url: URL of the user's uploaded document
        template_urls: List of template dictionaries with file_url and variation info
        
    Returns:
        Best matching template dictionary or None if no match found
    """
    try:
        if not template_urls:
            return None
        
        log.debug(f"Converting user document and {len(template_urls)} templates to images for analysis")
        
        # Convert user's PDF to image
        user_image_base64 = _convert_pdf_to_image(user_upload_url)
        if not user_image_base64:
            log.error("Failed to convert user PDF to image")
            return None
        
        # Convert template PDFs to images
        template_pdf_urls = [template['file_url'] for template in template_urls]
        template_images = _convert_pdfs_to_images(template_pdf_urls)
        
        # Filter out failed conversions
        valid_templates = []
        valid_images = []
        for i, (template, image) in enumerate(zip(template_urls, template_images)):
            if image:
                valid_templates.append(template)
                valid_images.append(image)
            else:
                log.warning(f"Failed to convert template {template['variation']} to image")
        
        if not valid_templates:
            log.error("No template images were successfully converted")
            return None
        
        log.debug(f"Successfully converted {len(valid_templates)} templates to images")
        
        # Create template information for prompt
        template_info = "\n".join([
            f"{i+1}. {template['variation']}"
            for i, template in enumerate(valid_templates)
        ])
        
        prompt = f"""
You are an expert PSA birth certificate layout analyzer. You will see the user's document image first, followed by template document images to compare against.

TEMPLATE OPTIONS TO COMPARE:
{template_info}

CRITICAL VISUAL ANALYSIS TASK:
Compare the user's document layout with each template document image to find the exact visual match.

DETAILED VISUAL COMPARISON CRITERIA:

1958 PSA Birth Certificate Visual Features:
- HEADER: "CERTIFICATE OF LIVE BIRTH" appears smaller, less prominent, basic typography
- LAYOUT: Simpler, more linear field arrangement with basic structure
- REGISTER NUMBER: Simple positioning at top, basic formatting style
- PARENT INFO: Arranged in straightforward linear rows, less complex subdivisions
- BIRTH DETAILS: Basic tabular format, fewer categories, simpler organization
- BORDERS/FIELDS: Simple field borders, less visual complexity and decoration
- OVERALL STYLE: Vintage document appearance, basic typography, compact design
- FORM STRUCTURE: Fewer distinct sections, more traditional form organization

1993 PSA Birth Certificate Visual Features:
- HEADER: "CERTIFICATE OF LIVE BIRTH" appears larger, more prominent, modern typography
- LAYOUT: Complex, multi-sectioned arrangement with sophisticated organization
- REGISTER NUMBER: More integrated into overall modern design layout
- PARENT INFO: Organized in structured boxes/sections with clear visual divisions
- BIRTH DETAILS: Detailed sub-fields, multiple categories, complex organization
- BORDERS/FIELDS: Modern field borders, sophisticated visual organization and styling
- OVERALL STYLE: Contemporary document styling, improved typography, expanded design
- FORM STRUCTURE: Multiple distinct sections with clear separators and modern layout

ANALYSIS INSTRUCTIONS:
1. Examine the user's document image carefully
2. Compare it with each template document image shown
3. Focus on the key visual differences:
   - Header size, style, and prominence level
   - Field organization complexity (simple vs complex)
   - Section divisions and visual structure patterns
   - Border styles and field arrangement
   - Overall document design (vintage 1958 vs modern 1993)
   - Typography styles and spacing patterns

4. Determine which template image has the most similar visual layout structure
5. Pay special attention to distinguishing 1958 (simpler/vintage) vs 1993 (complex/modern) designs

RESPONSE REQUIREMENT:
Return ONLY the exact variation name of the best matching template.
Examples: "English_1958_template" or "English_1993_template"
If no good visual match: "NO_MATCH"

Analyze the images now:
"""

        # Prepare content parts with images
        content_parts = [{"text": prompt}]
        
        # Add user document image
        content_parts.append({
            "text": "USER'S DOCUMENT TO ANALYZE:"
        })
        content_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": user_image_base64
            }
        })
        
        # Add template images
        for i, (template, image_base64) in enumerate(zip(valid_templates, valid_images)):
            content_parts.append({
                "text": f"\nTEMPLATE {i+1}: {template['variation']}"
            })
            content_parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_base64
                }
            })

        # Prepare the API request
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": content_parts
            }],
            "generationConfig": {
                "temperature": 0.01,
                "maxOutputTokens": 30,
                "topP": 0.9,
                "topK": 3
            }
        }
        
        # Make the API call with retries
        for attempt in range(RETRIES):
            try:
                response = requests.post(
                    f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=data,
                    timeout=TIMEOUT_SECONDS * 2  # Increase timeout for image processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("candidates") and len(result["candidates"]) > 0:
                        content = result["candidates"][0].get("content", {})
                        if content.get("parts") and len(content["parts"]) > 0:
                            llm_response = content["parts"][0].get("text", "").strip()
                            
                            log.debug(f"LLM image layout comparison response: {llm_response}")
                            
                            if llm_response == "NO_MATCH":
                                return None
                            
                            # Find the matching template
                            for template in valid_templates:
                                if template["variation"] == llm_response:
                                    log.debug(f"LLM found matching template: {llm_response}")
                                    return template
                            
                            log.warning(f"LLM returned variation not in template list: {llm_response}")
                            return None
                        
                else:
                    log.warning(f"Gemini API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt == RETRIES - 1:
                        return None
                    
            except requests.exceptions.Timeout:
                log.warning(f"Gemini API timeout (attempt {attempt + 1})")
                if attempt == RETRIES - 1:
                    return None
            except Exception as e:
                log.warning(f"Gemini API request error (attempt {attempt + 1}): {str(e)}")
                if attempt == RETRIES - 1:
                    return None
        
        return None
        
    except Exception as e:
        log.error(f"Error in LLM image layout comparison: {str(e)}")
        return None

def _find_translated_template_by_year(
    templates: List[Dict[str, Any]], 
    target_language: str, 
    year_variation: str
) -> Optional[Dict[str, Any]]:
    """
    Find the translated template with the same year variation.
    
    Args:
        templates: All available templates
        target_language: Target language for translation
        year_variation: Year part extracted from original template (e.g., "1993", "1958")
        
    Returns:
        Matching translated template or None
    """
    normalized_target_language = _normalize_language_for_template(target_language)
    
    for template in templates:
        variation = template.get("variation", "")
        if variation:
            # Check if this template matches target language and year
            parts = variation.split("_")
            if len(parts) >= 2:
                template_language = parts[0]
                template_year = parts[1]
                
                if (template_language.lower() == normalized_target_language.lower() and 
                    template_year == year_variation):
                    log.debug(f"Found translated template: {variation}")
                    return template
    
    log.warning(f"No translated template found for {normalized_target_language}_{year_variation}")
    return None

async def find_template_match_with_llm(
    latest_templatable_upload: Upload,
    translate_from: str,
    translate_to: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Find matching original and translated templates using LLM-based layout analysis.
    
    Args:
        latest_templatable_upload: The analyzed document upload
        translate_from: Source language of the document
        translate_to: Target language for translation
        
    Returns:
        Tuple of (original_template, translated_template) dictionaries or None if not found
    """
    try:
        log.debug(f"Finding template match with LLM for languages: {translate_from} -> {translate_to}")
        
        # Load sample templates
        all_templates = _load_sample_templates()
        if not all_templates:
            log.error("No sample templates available")
            return None, None
        
        # Filter templates by source language
        source_language_templates = _filter_templates_by_language(all_templates, translate_from)
        if not source_language_templates:
            log.warning(f"No templates found for source language: {translate_from}")
            return None, None
        
        # Use LLM to compare layouts and find the best match
        user_upload_url = latest_templatable_upload.public_url
        if not user_upload_url:
            log.error("No public URL available for user upload")
            return None, None
        
        log.debug(f"Comparing user upload {user_upload_url} with {len(source_language_templates)} templates")
        matching_template = await _llm_compare_layouts(user_upload_url, source_language_templates)
        
        if not matching_template:
            log.warning("LLM found no matching template for user upload")
            return None, None
        
        original_template = matching_template
        log.debug(f"LLM identified original template: {original_template['variation']}")
        
        # Extract year variation from the matched template
        variation_parts = original_template["variation"].split("_")
        if len(variation_parts) < 2:
            log.error(f"Invalid template variation format: {original_template['variation']}")
            return original_template, None
        
        year_variation = variation_parts[1]  # e.g., "1993", "1958"
        log.debug(f"Extracted year variation: {year_variation}")
        
        # Find the corresponding translated template with the same year
        translated_template = _find_translated_template_by_year(
            all_templates, translate_to, year_variation
        )
        
        if not translated_template:
            log.warning(f"No translated template found for {translate_to} with year {year_variation}")
            # If same language, use the same template
            if translate_from.lower() == translate_to.lower():
                translated_template = original_template
                log.debug("Source and target languages are the same, using same template")
        
        log.debug("LLM-based template matching completed successfully")
        return original_template, translated_template
        
    except Exception as e:
        log.error(f"Error in LLM-based template matching: {str(e)}")
        log.exception("LLM template matching failed")
        return None, None

async def find_template_match(
    db_client,
    latest_templatable_upload: Upload,
    translate_from: str,
    translate_to: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Find matching original and translated templates for the given document.
    
    This function first tries LLM-based matching, then falls back to the original heuristic approach.
    
    Args:
        db_client: Database client for Supabase queries
        latest_templatable_upload: The analyzed document upload
        translate_from: Source language of the document
        translate_to: Target language for translation
        
    Returns:
        Tuple of (original_template, translated_template) dictionaries or None if not found
    """
    try:
        log.debug(f"Finding template match for document type and languages: {translate_from} -> {translate_to}")
        
        # First, try LLM-based matching
        original_template, translated_template = await find_template_match_with_llm(
            latest_templatable_upload, translate_from, translate_to
        )
        
        if original_template:
            log.debug("LLM-based template matching succeeded")
            return original_template, translated_template
        
        log.debug("LLM-based matching failed, falling back to heuristic approach")
        
        # Fallback to original heuristic approach
        # Determine document type for template matching
        doc_type = _determine_doc_type_for_template(latest_templatable_upload)
        log.debug(f"Determined doc_type: {doc_type}")
        
        # Get format year from document analysis
        format_year = latest_templatable_upload.analysis.get("format_year", "unknown") if latest_templatable_upload.analysis else "unknown"
        log.debug(f"Document format year: {format_year}")
        
        # Check for reissued document indicators
        has_modern_security = latest_templatable_upload.analysis.get("has_psa_features", False) if latest_templatable_upload.analysis else False
        is_psa_document = latest_templatable_upload.analysis.get("is_psa_document", False) if latest_templatable_upload.analysis else False
        
        if format_year == "1958" and has_modern_security:
            log.debug("⚠️ Detected potentially reissued 1958 document with modern security features")
        elif format_year == "1993" and has_modern_security:
            log.debug("✅ Detected native 1993 document with expected security features")
        elif format_year == "unknown" and has_modern_security:
            log.debug("⚠️ Document has modern security features but format year is unknown")
        
        # Normalize languages for template matching
        original_language = _normalize_language_for_template(translate_from)
        target_language = _normalize_language_for_template(translate_to)
        
        log.debug(f"Normalized languages - From: {original_language}, To: {target_language}")
        
        # Query all templates for the document type
        templates_response = db_client.client.table("templates").select(
            "id, doc_type, variation, file_url, info_json, created_at"
        ).eq("doc_type", doc_type).execute()
        
        if not templates_response.data:
            log.warning(f"No templates found for doc_type: {doc_type}")
            return None, None
        
        templates = templates_response.data
        log.debug(f"Found {len(templates)} templates for doc_type: {doc_type}")
        
        # Find original template (source language + format year)
        original_template = None
        translated_template = None
        
        # Helper function to check if a template matches language and format year
        def template_matches(template_variation: str, language: str, year: str) -> bool:
            """Check if template variation matches language and format year."""
            variation_lower = template_variation.lower()
            language_lower = language.lower()
            
            # Check if language matches
            language_match = language_lower in variation_lower
            
            # Check if format year matches (if known)
            if year != "unknown":
                year_match = year in variation_lower
                return language_match and year_match
            else:
                # If format year is unknown, just match by language
                return language_match
        
        # Look for exact matches first (language + format year)
        log.debug("Looking for exact matches (language + format year)")
        for template in templates:
            variation = template.get("variation", "")
            
            # Check for original language match
            if template_matches(variation, original_language, format_year):
                original_template = template
                log.debug(f"Found exact original template: {template['id']} - {variation}")
            
            # Check for target language match
            if template_matches(variation, target_language, format_year):
                translated_template = template
                log.debug(f"Found exact translated template: {template['id']} - {variation}")
        
        # If we couldn't find exact matches and format year is known, try language-only matches
        if (not original_template or not translated_template) and format_year != "unknown":
            log.debug("Exact matches not found, trying language-only matches")
            
            for template in templates:
                variation = template.get("variation", "")
                
                # Check for original language match (without format year requirement)
                if not original_template and original_language.lower() in variation.lower():
                    original_template = template
                    log.debug(f"Found language-only original template: {template['id']} - {variation}")
                
                # Check for target language match (without format year requirement)
                if not translated_template and target_language.lower() in variation.lower():
                    translated_template = template
                    log.debug(f"Found language-only translated template: {template['id']} - {variation}")
        
        # Final fallback: English templates
        if not original_template:
            log.debug("No original template found, trying English fallback")
            for template in templates:
                variation = template.get("variation", "")
                if "english" in variation.lower():
                    # Prefer English template with matching format year if available
                    if format_year != "unknown" and format_year in variation.lower():
                        original_template = template
                        log.debug(f"Found English fallback with format year: {template['id']} - {variation}")
                        break
                    elif not original_template:  # Keep first English template as backup
                        original_template = template
                        log.debug(f"Found English fallback template: {template['id']} - {variation}")
        
        # If both languages are the same, use the same template for both
        if original_language == target_language and original_template:
            translated_template = original_template
            log.debug("Source and target languages are the same, using same template for both")
        
        # Validate that we have at least an original template
        if not original_template:
            log.error(f"Could not find any suitable template for doc_type: {doc_type}, language: {original_language}, format_year: {format_year}")
            return None, None
        
        # Log template selection details
        if original_template:
            original_info = original_template.get("info_json", {})
            log.debug(f"Selected original template: {original_template['variation']} with {len(original_info.get('required_fields', {}))} required fields")
        
        if translated_template:
            translated_info = translated_template.get("info_json", {})
            log.debug(f"Selected translated template: {translated_template['variation']} with {len(translated_info.get('required_fields', {}))} required fields")
        else:
            log.debug(f"No specific translated template found for {target_language}, will use dynamic translation")
        
        log.debug("Heuristic template matching completed successfully")
        return original_template, translated_template
        
    except Exception as e:
        log.error(f"Error finding template match: {str(e)}")
        log.exception("Template matching failed")
        return None, None

def validate_template_compatibility(template: Dict[str, Any], upload: Upload) -> bool:
    """
    Validate that a template is compatible with the uploaded document.
    
    Args:
        template: Template data from database
        upload: Upload object with analysis data
        
    Returns:
        Boolean indicating if template is compatible
    """
    try:
        if not template or not upload.analysis:
            return False
        
        # Check if document is identified as templatable
        if not upload.is_templatable:
            log.debug("Document is not marked as templatable")
            return False
        
        # Check confidence level from analysis
        confidence = upload.analysis.get("confidence", 0.0)
        if confidence < 0.6:
            log.debug(f"Document analysis confidence too low: {confidence}")
            return False
        
        # Check if it's a PSA document (our templates are PSA-specific)
        is_psa = upload.analysis.get("is_psa_document", False)
        has_psa_features = upload.analysis.get("has_psa_features", False)
        
        if not (is_psa or has_psa_features):
            log.debug("Document doesn't appear to be a PSA document")
            return False
        
        # Check page count compatibility
        page_count = upload.analysis.get("page_count")
        if page_count and page_count > 3:  # PSA documents are typically 1-2 pages
            log.debug(f"Document has too many pages for PSA template: {page_count}")
            return False
        
        # Check format year compatibility
        doc_format_year = upload.analysis.get("format_year", "unknown")
        template_variation = template.get("variation", "")
        
        if doc_format_year != "unknown":
            # If document has a known format year, prefer templates with matching format year
            if doc_format_year in template_variation:
                log.debug(f"Template format year matches document: {doc_format_year}")
            else:
                log.debug(f"Template format year mismatch - Document: {doc_format_year}, Template: {template_variation}")
                # Don't fail validation but log the mismatch - the system can still work with mismatched years
        
        # Additional validation for reissued documents
        # Check if document has clear PSA features regardless of format year
        if upload.analysis.get("is_psa_document", False):
            log.debug("Document confirmed as PSA document - template compatibility maintained")
        
        # For reissued documents, we need to be more flexible about security features
        # The presence of modern security features doesn't disqualify older format templates
        has_modern_security = upload.analysis.get("has_psa_features", False)
        if has_modern_security and doc_format_year == "1958":
            log.debug("Document appears to be reissued 1958 format with modern security features")
        elif has_modern_security and doc_format_year == "1993":
            log.debug("Document appears to be native 1993 format with expected security features")
        
        log.debug("Template compatibility validation passed")
        return True
        
    except Exception as e:
        log.error(f"Error validating template compatibility: {str(e)}")
        return False

def _convert_pdf_to_image(pdf_url: str) -> Optional[str]:
    """
    Convert PDF from URL to base64 encoded image for LLM analysis.
    
    Args:
        pdf_url: URL of the PDF document
        
    Returns:
        Base64 encoded image string or None if conversion fails
    """
    try:
        log.debug(f"Converting PDF to image: {pdf_url}")
        
        # Download PDF from URL
        response = requests.get(pdf_url, timeout=30)
        if response.status_code != 200:
            log.error(f"Failed to download PDF: {response.status_code}")
            return None
        
        pdf_bytes = response.content
        log.debug(f"Downloaded PDF: {len(pdf_bytes)} bytes")
        
        # Convert PDF bytes to images (get first page only)
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200, fmt='jpeg')
        
        if not images:
            log.warning(f"No images generated from PDF: {pdf_url}")
            return None
        
        # Get the first page
        image = images[0]
        
        # Resize image if too large (max 1024x1024 for better LLM processing)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / image.width, max_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        log.debug(f"Successfully converted PDF to image: {len(image_base64)} characters")
        return image_base64
        
    except Exception as e:
        log.error(f"Error converting PDF to image: {str(e)}")
        return None

def _convert_pdfs_to_images(pdf_urls: List[str]) -> List[Optional[str]]:
    """
    Convert multiple PDF URLs to base64 encoded images.
    
    Args:
        pdf_urls: List of PDF URLs
        
    Returns:
        List of base64 encoded image strings (None for failed conversions)
    """
    images = []
    for pdf_url in pdf_urls:
        image_base64 = _convert_pdf_to_image(pdf_url)
        images.append(image_base64)
    return images