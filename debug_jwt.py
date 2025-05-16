"""
Debug script to validate JWT token with different decoding methods.
Save this as debug_jwt.py in your project root.
"""

import sys
import jwt
import base64
from typing import Dict, Any, Optional

def decode_token_with_options(
    token: str, 
    secret: str, 
    algorithms: list = ["HS256"], 
    options: Dict[str, Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Try to decode a JWT token with given options.
    
    Args:
        token: The JWT token to decode
        secret: The secret to use for verification
        algorithms: List of algorithms to try
        options: JWT decode options
        
    Returns:
        Dict or None: The decoded token payload or None if decoding fails
    """
    if options is None:
        options = {
            'verify_signature': True,
            'verify_aud': False,
            'verify_iat': True,
            'verify_exp': True,
            'verify_nbf': False,
            'verify_iss': False,
            'verify_sub': False,
            'verify_jti': False,
            'verify_at_hash': False,
            'require_aud': False,
            'require_iat': False,
            'require_exp': False,
            'require_nbf': False,
            'require_iss': False,
            'require_sub': False,
            'require_jti': False,
            'require_at_hash': False,
            'leeway': 0,
        }
    
    try:
        payload = jwt.decode(token, secret, algorithms=algorithms, options=options)
        return payload
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def main():
    """Main function to debug JWT token."""
    if len(sys.argv) < 3:
        print("Usage: python debug_jwt.py <token> <secret>")
        sys.exit(1)
    
    token = sys.argv[1]
    secret = sys.argv[2]
    
    print("\n=== JWT Token Debug ===\n")
    
    # Display token info
    print(f"Token (first 10 chars): {token[:10]}...")
    
    # Try to decode without verification
    try:
        header = jwt.get_unverified_header(token)
        print(f"\nToken Header: {header}")
        
        payload = jwt.decode(token, options={"verify_signature": False})
        print(f"\nUnverified Payload: {payload}")
        print(f"\nPayload Keys: {list(payload.keys())}")
        
        # Try to find user identification
        user_id = payload.get("sub") or payload.get("user_id") or payload.get("id")
        email = payload.get("email")
        print(f"\nUser ID: {user_id}")
        print(f"Email: {email}")
        
    except Exception as e:
        print(f"Error decoding token without verification: {str(e)}")
    
    # Try various decoding methods
    print("\n=== Verification Methods ===\n")
    
    print("1. Raw Secret:")
    result = decode_token_with_options(token, secret)
    if result:
        print("  Success! Token verified with raw secret")
    
    print("\n2. Padded Secret:")
    padded_secret = secret
    missing_padding = len(secret) % 4
    if missing_padding:
        padded_secret += '=' * (4 - missing_padding)
        
    print(f"  Padded secret length: {len(padded_secret)}")
    result = decode_token_with_options(token, padded_secret)
    if result:
        print("  Success! Token verified with padded secret")
    
    print("\n3. Base64 Decoded Secret:")
    try:
        decoded_secret = base64.b64decode(padded_secret)
        print(f"  Decoded secret length: {len(decoded_secret)}")
        result = decode_token_with_options(token, decoded_secret)
        if result:
            print("  Success! Token verified with base64 decoded secret")
    except Exception as e:
        print(f"  Error decoding secret: {str(e)}")
    
    print("\n4. Using RS256 Algorithm:")
    result = decode_token_with_options(token, secret, algorithms=["RS256"])
    if result:
        print("  Success! Token verified with RS256 algorithm")
    
    print("\n5. Using Token's Own Algorithm:")
    alg = header.get('alg', 'HS256')
    result = decode_token_with_options(token, secret, algorithms=[alg])
    if result:
        print(f"  Success! Token verified with {alg} algorithm")

    print("\n=== Recommendation ===\n")
    if not any([result]):
        print("None of the methods worked with the provided secret.")
        print("Possible solutions:")
        print("1. Check if the JWT_SECRET in your .env file matches the Supabase project's JWT secret")
        print("2. Try disabling signature verification temporarily for testing")
        print("3. Check with Supabase documentation for the correct way to retrieve the JWT secret")
        print("4. Consider using Supabase's built-in JWT verification functions if available")
    else:
        print("At least one method worked! Check the results above for the successful method.")


if __name__ == "__main__":
    main()