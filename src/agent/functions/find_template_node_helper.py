# backend/src/agent/functions/find_template_node_helper.py
"""
Template matching helper for finding original and translated template pairs.

This module provides the `find_template_match` function that:
1. Analyzes the uploaded document's characteristics
2. Finds the appropriate original template based on document type and source language
3. Finds the corresponding translated template for the target language
4. Returns structured template information for workflow processing

The function searches the Supabase templates table and matches templates based on:
- Document type (e.g., birth_certificate)
- Language variations (source and target languages)
- Template compatibility with the analyzed document
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional
from ..agent_state import AgentState, Upload

log = logging.getLogger(__name__)

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

async def find_template_match(
    db_client,
    latest_templatable_upload: Upload,
    translate_from: str,
    translate_to: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Find matching original and translated templates for the given document.
    
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
        
        # Determine document type for template matching
        doc_type = _determine_doc_type_for_template(latest_templatable_upload)
        log.debug(f"Determined doc_type: {doc_type}")
        
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
        
        # Find original template (source language)
        original_template = None
        translated_template = None
        
        # Look for templates matching the language variations
        for template in templates:
            variation = template.get("variation", "")
            
            # Check if this template matches the original language
            if original_language.lower() in variation.lower():
                original_template = template
                log.debug(f"Found original template: {template['id']} - {variation}")
            
            # Check if this template matches the target language
            if target_language.lower() in variation.lower():
                translated_template = template
                log.debug(f"Found translated template: {template['id']} - {variation}")
        
        # If we couldn't find exact matches, try fallback logic
        if not original_template:
            # Default to English template as fallback for original
            for template in templates:
                if "english" in template.get("variation", "").lower():
                    original_template = template
                    log.debug(f"Using English fallback for original: {template['id']}")
                    break
        
        if not translated_template and target_language != "English":
            # If no specific translated version exists, we'll use the original for now
            # The system can handle translation without a specific template
            log.debug(f"No specific template found for {target_language}, will rely on dynamic translation")
        
        # If both languages are the same, use the same template for both
        if original_language == target_language and original_template:
            translated_template = original_template
            log.debug("Source and target languages are the same, using same template for both")
        
        # Validate that we have at least an original template
        if not original_template:
            log.error(f"Could not find any suitable template for doc_type: {doc_type}")
            return None, None
        
        # Extract additional info from templates
        if original_template:
            original_info = original_template.get("info_json", {})
            log.debug(f"Original template required fields: {len(original_info.get('required_fields', {}))}")
        
        if translated_template:
            translated_info = translated_template.get("info_json", {})
            log.debug(f"Translated template required fields: {len(translated_info.get('required_fields', {}))}")
        
        log.debug("Template matching completed successfully")
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
        
        log.debug("Template compatibility validation passed")
        return True
        
    except Exception as e:
        log.error(f"Error validating template compatibility: {str(e)}")
        return False