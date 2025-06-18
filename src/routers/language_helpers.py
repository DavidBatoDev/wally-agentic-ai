"""
Language Helper Functions
Helper functions for language code normalization and validation for translation services.
"""

from typing import Tuple, Optional


def normalize_language_code(language: str) -> str:
    """
    Convert language names to ISO 639-1 codes for Google Translate API.
    If already normalized (valid ISO code), returns as-is.
    If not normalized, converts full language names to ISO codes.
    
    Args:
        language (str): Language name or code to normalize
        
    Returns:
        str: Normalized ISO 639-1 language code
        
    Examples:
        >>> normalize_language_code("Greek")
        'el'
        >>> normalize_language_code("en")
        'en'
        >>> normalize_language_code("Español")
        'es'
    """
    if not language:
        return language
    
    # Clean the input: strip whitespace and convert to lowercase for comparison
    language_clean = language.strip().lower()
    
    # Valid ISO 639-1 codes that Google Translate supports
    valid_iso_codes = {
        'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 
        'ceb', 'ny', 'zh', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'tl', 
        'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'he', 
        'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 
        'km', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 
        'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ps', 'fa', 'pl', 'pt', 
        'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 
        'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 
        'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu'
    }
    
    # If it's already a valid ISO code, return it
    if language_clean in valid_iso_codes:
        return language_clean
    
    # Language name to ISO code mapping
    language_mapping = {
        # Major languages with common variations
        'english': 'en',
        'spanish': 'es', 
        'espanol': 'es',
        'español': 'es',
        'castilian': 'es',
        'french': 'fr',
        'francais': 'fr',
        'français': 'fr',
        'german': 'de',
        'deutsch': 'de',
        'italian': 'it',
        'italiano': 'it',
        'portuguese': 'pt',
        'português': 'pt',
        'brazilian portuguese': 'pt',
        'european portuguese': 'pt',
        'russian': 'ru',
        'русский': 'ru',
        'chinese': 'zh',
        'mandarin': 'zh',
        'simplified chinese': 'zh',
        'traditional chinese': 'zh',
        '中文': 'zh',
        'japanese': 'ja',
        '日本語': 'ja',
        'korean': 'ko',
        '한국어': 'ko',
        'arabic': 'ar',
        'العربية': 'ar',
        'hindi': 'hi',
        'हिन्दी': 'hi',
        'dutch': 'nl',
        'nederlands': 'nl',
        'swedish': 'sv',
        'svenska': 'sv',
        'norwegian': 'no',
        'norsk': 'no',
        'danish': 'da',
        'dansk': 'da',
        'finnish': 'fi',
        'suomi': 'fi',
        'greek': 'el',
        'ελληνικά': 'el',
        
        # Additional European languages
        'polish': 'pl',
        'polski': 'pl',
        'czech': 'cs',
        'čeština': 'cs',
        'hungarian': 'hu',
        'magyar': 'hu',
        'romanian': 'ro',
        'română': 'ro',
        'bulgarian': 'bg',
        'български': 'bg',
        'croatian': 'hr',
        'hrvatski': 'hr',
        'serbian': 'sr',
        'српски': 'sr',
        'slovak': 'sk',
        'slovenčina': 'sk',
        'slovenian': 'sl',
        'slovenščina': 'sl',
        'lithuanian': 'lt',
        'lietuvių': 'lt',
        'latvian': 'lv',
        'latviešu': 'lv',
        'estonian': 'et',
        'eesti': 'et',
        'ukrainian': 'uk',
        'українська': 'uk',
        
        # Asian languages
        'thai': 'th',
        'ไทย': 'th',
        'vietnamese': 'vi',
        'tiếng việt': 'vi',
        'indonesian': 'id',
        'bahasa indonesia': 'id',
        'malay': 'ms',
        'bahasa melayu': 'ms',
        'tagalog': 'tl',
        'filipino': 'tl',
        'bengali': 'bn',
        'বাংলা': 'bn',
        'urdu': 'ur',
        'اردو': 'ur',
        'persian': 'fa',
        'farsi': 'fa',
        'فارسی': 'fa',
        'hebrew': 'he',
        'עברית': 'he',
        'turkish': 'tr',
        'türkçe': 'tr',
        
        # African languages
        'swahili': 'sw',
        'kiswahili': 'sw',
        'afrikaans': 'af',
        'amharic': 'am',
        'አማርኛ': 'am',
        'yoruba': 'yo',
        'igbo': 'ig',
        'hausa': 'ha',
        'zulu': 'zu',
        'xhosa': 'xh',
        
        # Other languages
        'esperanto': 'eo',
        'latin': 'la',
        'welsh': 'cy',
        'cymraeg': 'cy',
        'irish': 'ga',
        'gaeilge': 'ga',
        'scottish gaelic': 'gd',
        'gàidhlig': 'gd',
        'basque': 'eu',
        'euskera': 'eu',
        'catalan': 'ca',
        'català': 'ca',
        'galician': 'gl',
        'galego': 'gl',
        'maltese': 'mt',
        'malti': 'mt',
        'icelandic': 'is',
        'íslenska': 'is',
        
        # Common English variations
        'american english': 'en',
        'british english': 'en',
        'australian english': 'en',
    }
    
    # Try to find a match in the mapping
    normalized_code = language_mapping.get(language_clean)
    
    if normalized_code:
        return normalized_code
    
    # Handle language variants (e.g., 'zh-cn', 'en-us', 'pt_BR')
    if '-' in language_clean:
        main_code = language_clean.split('-')[0]
        if main_code in valid_iso_codes:
            return main_code
    
    if '_' in language_clean:
        main_code = language_clean.split('_')[0]
        if main_code in valid_iso_codes:
            return main_code
    
    # If no match found, return lowercase version with warning
    print(f"Warning: Unknown language '{language}', using as-is")
    return language.lower()


def validate_language_code(language_code: str) -> bool:
    """
    Validate if a language code is supported by Google Translate API.
    
    Args:
        language_code (str): ISO 639-1 language code
        
    Returns:
        bool: True if the language code is valid/supported
        
    Examples:
        >>> validate_language_code("en")
        True
        >>> validate_language_code("xyz")
        False
    """
    if not language_code:
        return False
    
    # Valid ISO 639-1 codes that Google Translate supports
    valid_iso_codes = {
        'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 
        'ceb', 'ny', 'zh', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et', 'tl', 
        'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'he', 
        'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 
        'km', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms', 
        'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ps', 'fa', 'pl', 'pt', 
        'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 
        'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 
        'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu'
    }
    
    return language_code.lower().strip() in valid_iso_codes


def normalize_and_validate_language(language: str) -> Tuple[str, bool]:
    """
    Normalize a language and validate if it's supported.
    
    Args:
        language (str): Language name or code to normalize
        
    Returns:
        Tuple[str, bool]: (normalized_code, is_valid)
        
    Examples:
        >>> normalize_and_validate_language("Greek")
        ('el', True)
        >>> normalize_and_validate_language("Unknown Language")
        ('unknown language', False)
    """
    normalized = normalize_language_code(language)
    is_valid = validate_language_code(normalized)
    return normalized, is_valid


def get_language_name(language_code: str) -> Optional[str]:
    """
    Get the English name of a language from its ISO code.
    
    Args:
        language_code (str): ISO 639-1 language code
        
    Returns:
        Optional[str]: English name of the language, or None if not found
        
    Examples:
        >>> get_language_name("el")
        'Greek'
        >>> get_language_name("en")
        'English'
    """
    if not language_code:
        return None
    
    code_to_name = {
        'af': 'Afrikaans',
        'sq': 'Albanian',
        'am': 'Amharic',
        'ar': 'Arabic',
        'hy': 'Armenian',
        'az': 'Azerbaijani',
        'eu': 'Basque',
        'be': 'Belarusian',
        'bn': 'Bengali',
        'bs': 'Bosnian',
        'bg': 'Bulgarian',
        'ca': 'Catalan',
        'ceb': 'Cebuano',
        'ny': 'Chichewa',
        'zh': 'Chinese',
        'co': 'Corsican',
        'hr': 'Croatian',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'eo': 'Esperanto',
        'et': 'Estonian',
        'tl': 'Filipino',
        'fi': 'Finnish',
        'fr': 'French',
        'fy': 'Frisian',
        'gl': 'Galician',
        'ka': 'Georgian',
        'de': 'German',
        'el': 'Greek',
        'gu': 'Gujarati',
        'ht': 'Haitian Creole',
        'ha': 'Hausa',
        'haw': 'Hawaiian',
        'he': 'Hebrew',
        'hi': 'Hindi',
        'hmn': 'Hmong',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'ig': 'Igbo',
        'id': 'Indonesian',
        'ga': 'Irish',
        'it': 'Italian',
        'ja': 'Japanese',
        'jw': 'Javanese',
        'kn': 'Kannada',
        'kk': 'Kazakh',
        'km': 'Khmer',
        'ko': 'Korean',
        'ku': 'Kurdish',
        'ky': 'Kyrgyz',
        'lo': 'Lao',
        'la': 'Latin',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'lb': 'Luxembourgish',
        'mk': 'Macedonian',
        'mg': 'Malagasy',
        'ms': 'Malay',
        'ml': 'Malayalam',
        'mt': 'Maltese',
        'mi': 'Maori',
        'mr': 'Marathi',
        'mn': 'Mongolian',
        'my': 'Myanmar',
        'ne': 'Nepali',
        'no': 'Norwegian',
        'ps': 'Pashto',
        'fa': 'Persian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'pa': 'Punjabi',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sm': 'Samoan',
        'gd': 'Scottish Gaelic',
        'sr': 'Serbian',
        'st': 'Sesotho',
        'sn': 'Shona',
        'sd': 'Sindhi',
        'si': 'Sinhala',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'so': 'Somali',
        'es': 'Spanish',
        'su': 'Sundanese',
        'sw': 'Swahili',
        'sv': 'Swedish',
        'tg': 'Tajik',
        'ta': 'Tamil',
        'te': 'Telugu',
        'th': 'Thai',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'uz': 'Uzbek',
        'vi': 'Vietnamese',
        'cy': 'Welsh',
        'xh': 'Xhosa',
        'yi': 'Yiddish',
        'yo': 'Yoruba',
        'zu': 'Zulu'
    }
    
    return code_to_name.get(language_code.lower().strip())


# Example usage and tests
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Greek", "greek", "GREEK", "ελληνικά",
        "English", "english", "en", "EN", "American English",
        "Spanish", "Español", "es", "Castilian",
        "Chinese", "zh", "Mandarin", "中文", "Simplified Chinese",
        "Portuguese", "pt-BR", "Brazilian Portuguese",
        "French", "français", "fr-CA",
        "Unknown Language", "xyz", ""
    ]
    
    print("Testing language normalization:")
    print("-" * 60)
    for test_lang in test_cases:
        normalized = normalize_language_code(test_lang)
        is_valid = validate_language_code(normalized)
        language_name = get_language_name(normalized)
        print(f"'{test_lang}' -> '{normalized}' (valid: {is_valid}) ({language_name})")