"""
Simple FastAPI test for Google Translate API using REST API directly.
Test translation of "David" to Greek.
"""

import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Debug: Print environment variable loading
print("🔍 Debug: Checking environment variables...")
api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
print(f"🔍 API Key loaded: {'✅ Yes' if api_key else '❌ No'}")
if api_key:
    print(f"🔍 API Key length: {len(api_key)}")
    print(f"🔍 API Key starts with: {api_key[:10]}...")

# Initialize FastAPI app
app = FastAPI(title="Translation Test API", version="1.0.0")

# Google Translate API endpoint
TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"
DETECT_API_URL = "https://translation.googleapis.com/language/translate/v2/detect"

def translate_text_api(text: str, target_language: str, source_language: str = None):
    """Translate text using Google Translate REST API."""
    if not api_key:
        raise ValueError("API key not configured")
    
    params = {
        'key': api_key,
        'q': text,
        'target': target_language
    }
    
    if source_language:
        params['source'] = source_language
    
    try:
        response = requests.post(TRANSLATE_API_URL, params=params)
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

def detect_language_api(text: str):
    """Detect language using Google Translate REST API."""
    if not api_key:
        raise ValueError("API key not configured")
    
    params = {
        'key': api_key,
        'q': text
    }
    
    try:
        response = requests.post(DETECT_API_URL, params=params)
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

# Test API connection on startup
translate_api_ready = False
try:
    print("🔄 Testing Google Translate API connection...")
    test_result = translate_text_api("Hello", "es")
    print(f"✅ Google Translate API test successful: '{test_result['translatedText']}'")
    translate_api_ready = True
except Exception as e:
    print(f"❌ Google Translate API test failed: {e}")
    translate_api_ready = False

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Translation Test API",
        "status": "running",
        "translate_api_ready": translate_api_ready,
        "api_key_configured": bool(api_key)
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check configuration."""
    return {
        "api_key_configured": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "translate_api_ready": translate_api_ready,
        "api_endpoints": {
            "translate": TRANSLATE_API_URL,
            "detect": DETECT_API_URL
        }
    }

@app.get("/test-david")
async def test_david_translation():
    """Test translating 'David' to Greek."""
    if not translate_api_ready:
        raise HTTPException(
            status_code=500, 
            detail="Google Translate API not ready. Check your API key and connection."
        )
    
    try:
        # Translate "David" to Greek
        text = "David"
        target_language = "el"  # Greek language code
        
        print(f"🔄 Translating '{text}' to {target_language}...")
        
        result = translate_text_api(text, target_language)
        
        response = {
            "original_text": text,
            "translated_text": result['translatedText'],
            "source_language": result.get('detectedSourceLanguage', 'en'),
            "target_language": target_language,
            "target_language_name": "Greek"
        }
        
        print(f"✅ Translation successful: {response}")
        return response
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/translate/{text}/{target_lang}")
async def translate_text(text: str, target_lang: str):
    """Translate any text to any target language."""
    if not translate_api_ready:
        raise HTTPException(
            status_code=500, 
            detail="Google Translate API not ready. Check your API key and connection."
        )
    
    try:
        print(f"🔄 Translating '{text}' to {target_lang}...")
        
        result = translate_text_api(text, target_lang)
        
        response = {
            "original_text": text,
            "translated_text": result['translatedText'],
            "source_language": result.get('detectedSourceLanguage', 'unknown'),
            "target_language": target_lang
        }
        
        print(f"✅ Translation successful: {response}")
        return response
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/detect-language/{text}")
async def detect_language(text: str):
    """Detect the language of given text."""
    if not translate_api_ready:
        raise HTTPException(
            status_code=500, 
            detail="Google Translate API not ready. Check your API key and connection."
        )
    
    try:
        print(f"🔄 Detecting language for: '{text}'...")
        
        result = detect_language_api(text)
        
        response = {
            "text": text,
            "detected_language": result['language'],
            "confidence": result.get('confidence', 0)
        }
        
        print(f"✅ Language detection successful: {response}")
        return response
        
    except Exception as e:
        print(f"❌ Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "translate_api_ready": translate_api_ready
    }

if __name__ == "__main__":
    print("🚀 Starting Translation Test API...")
    print("📋 Available endpoints:")
    print("   GET /                    - API info")
    print("   GET /debug               - Debug configuration info")
    print("   GET /test-david          - Test 'David' to Greek translation")
    print("   GET /translate/{text}/{target_lang} - Translate any text")
    print("   GET /detect-language/{text} - Detect text language")
    print("   GET /health              - Health check")
    print()
    print("🔗 Test the David translation at: http://localhost:8000/test-david")
    print("🔗 Debug info at: http://localhost:8000/debug")
    print("🔗 API docs at: http://localhost:8000/docs")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )