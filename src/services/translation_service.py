# backend/src/services/translation_service.py
"""
Translation service for workflow fields using Google Translate API and Gemini API.
"""

import json
import requests
from typing import Dict, Any, Optional, List, Tuple
import google.generativeai as genai
from ..config import get_settings

settings = get_settings()

class TranslationService:
    def __init__(self):
        # Google Translate API endpoints
        self.translate_api_url = "https://translation.googleapis.com/language/translate/v2"
        self.detect_api_url = "https://translation.googleapis.com/language/translate/v2/detect"
        
        # Check for API key
        if not settings.GOOGLE_TRANSLATE_API_KEY:
            raise ValueError("GOOGLE_TRANSLATE_API_KEY must be provided")
        
        self.api_key = settings.GOOGLE_TRANSLATE_API_KEY
        
        # Configure Gemini API
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            print("Warning: GEMINI_API_KEY not provided. Gemini translation will not be available.")
            self.gemini_model = None
    
    def _make_translate_request(self, text: str, target_language: str, source_language: str = None) -> Dict[str, Any]:
        """
        Make a request to Google Translate API.
        """
        params = {
            'key': self.api_key,
            'q': text,
            'target': target_language
        }
        
        if source_language:
            params['source'] = source_language
        
        try:
            response = requests.post(self.translate_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and 'translations' in data['data']:
                translation = data['data']['translations'][0]
                return {
                    'translatedText': translation['translatedText'],
                    'detectedSourceLanguage': translation.get('detectedSourceLanguage', source_language or 'unknown')
                }
            else:
                raise Exception(f"Unexpected API response format: {data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except Exception as e:
            raise Exception(f"Translation failed: {e}")
    
    def _make_detect_request(self, text: str) -> Dict[str, Any]:
        """
        Make a request to Google Translate Language Detection API.
        """
        params = {
            'key': self.api_key,
            'q': text
        }
        
        try:
            response = requests.post(self.detect_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and 'detections' in data['data']:
                detection = data['data']['detections'][0][0]
                return {
                    'language': detection['language'],
                    'confidence': detection.get('confidence', 0)
                }
            else:
                raise Exception(f"Unexpected API response format: {data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except Exception as e:
            raise Exception(f"Language detection failed: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text using Google Translate API.
        Returns language code (e.g., 'en', 'es', 'fr').
        """
        try:
            if not text or not text.strip():
                return 'en'  # Default to English for empty text
            
            result = self._make_detect_request(text)
            return result['language']
        except Exception as e:
            print(f"Error detecting language: {e}")
            return 'en'  # Default to English on error
    
    def translate_with_google(self, text: str, target_language: str, source_language: str = None) -> str:
        """
        Translate text using Google Translate API.
        """
        try:
            if not text or not text.strip():
                return text
            
            result = self._make_translate_request(text, target_language, source_language)
            return result['translatedText']
        except Exception as e:
            print(f"Error translating with Google Translate: {e}")
            return text  # Return original text on error
    
    def translate_with_gemini(self, text: str, target_language: str, source_language: str = None, context: str = None) -> str:
        """
        Translate text using Gemini API with better context understanding.
        """
        try:
            if not text or not text.strip():
                return text
            
            if not self.gemini_model:
                print("Gemini model not available, falling back to Google Translate")
                return self.translate_with_google(text, target_language, source_language)
            
            # Build prompt for Gemini
            prompt_parts = [
                f"Translate the following text to {target_language}."
            ]
            
            if source_language:
                prompt_parts.append(f"The source language is {source_language}.")
            
            if context:
                prompt_parts.append(f"Context: This is a {context} field from a document form.")
            
            prompt_parts.extend([
                "Maintain the original formatting and structure.",
                "Only return the translated text, nothing else.",
                f"Text to translate: {text}"
            ])
            
            prompt = " ".join(prompt_parts)
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error translating with Gemini: {e}")
            # Fallback to Google Translate
            return self.translate_with_google(text, target_language, source_language)
    
    def translate_field_value(
        self, 
        value: str, 
        target_language: str, 
        source_language: str = None, 
        field_context: str = None,
        use_gemini: bool = True
    ) -> str:
        """
        Translate a single field value using the preferred translation method.
        """
        if use_gemini and self.gemini_model:
            return self.translate_with_gemini(value, target_language, source_language, field_context)
        else:
            return self.translate_with_google(value, target_language, source_language)
    
    def translate_multiple_fields(
        self, 
        fields: Dict[str, Any], 
        target_language: str, 
        source_language: str = None,
        use_gemini: bool = True
    ) -> Dict[str, str]:
        """
        Translate multiple field values and return a dictionary of translations.
        """
        translations = {}
        
        for field_key, field_value in fields.items():
            if isinstance(field_value, str) and field_value.strip():
                try:
                    translated_value = self.translate_field_value(
                        field_value, 
                        target_language, 
                        source_language, 
                        field_context=field_key,
                        use_gemini=use_gemini
                    )
                    translations[field_key] = translated_value
                except Exception as e:
                    print(f"Error translating field {field_key}: {e}")
                    translations[field_key] = field_value  # Keep original on error
            else:
                translations[field_key] = field_value  # Keep non-string values as is
        
        return translations
    
    def get_language_name(self, language_code: str) -> str:
        """
        Convert language code to human-readable name.
        """
        language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'el': 'Greek'
        }
        return language_names.get(language_code.lower(), language_code.upper())
    
    def get_language_code(self, language_name: str) -> str:
        """
        Convert human-readable language name to language code (e.g., "Greek" -> "el").
        This is case-insensitive and also accepts language codes.
        """
        language_map = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
            'ar': 'Arabic', 'hi': 'Hindi', 'th': 'Thai', 'vi': 'Vietnamese', 'nl': 'Dutch',
            'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'pl': 'Polish',
            'tr': 'Turkish', 'el': 'Greek'
        }

        # Create a lookup map that handles names and codes, all lowercase
        lookup_map = {name.lower(): code for code, name in language_map.items()}
        for code in language_map.keys():
            lookup_map[code] = code  # e.g., lookup_map['el'] = 'el'

        cleaned_input = language_name.strip().lower()

        # Look up the code; fallback to original input if not found
        return lookup_map.get(cleaned_input, language_name)
    
    def test_connection(self) -> bool:
        """
        Test the connection to Google Translate API.
        """
        try:
            test_result = self._make_translate_request("Hello", "es")
            print(f"✅ Google Translate API test successful: '{test_result['translatedText']}'")
            return True
        except Exception as e:
            print(f"❌ Google Translate API test failed: {e}")
            return False


# Create a global instance
translation_service = TranslationService()