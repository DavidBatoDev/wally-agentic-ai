import os
import requests
from pathlib import Path
import fitz

def download_working_font():
    """
    Download a single, reliable font that definitely works with Greek text.
    """
    print("=== Simple Font Download ===")
    
    # Create fonts directory
    fonts_dir = Path("fonts")
    fonts_dir.mkdir(exist_ok=True)
    
    # Reliable font URL - Noto Sans from Google Fonts GitHub
    font_info = {
        "name": "NotoSans-Regular.ttf",
        "url": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
    }
    
    font_path = fonts_dir / font_info["name"]
    
    # Skip if already exists
    if font_path.exists():
        print(f"✓ Font already exists: {font_path}")
        return test_font(font_path)
    
    try:
        print(f"Downloading {font_info['name']}...")
        
        # Download with headers to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(font_info["url"], headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save font file
        with open(font_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {font_info['name']} ({len(response.content)} bytes)")
        
        # Test the font
        return test_font(font_path)
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return try_backup_method()

def try_backup_method():
    """
    Backup method: Try multiple font sources.
    """
    print("\nTrying backup font sources...")
    
    backup_fonts = [
        {
            "name": "Roboto-Regular.ttf",
            "url": "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
        },
        {
            "name": "OpenSans-Regular.ttf", 
            "url": "https://github.com/googlefonts/opensans/raw/main/fonts/ttf/OpenSans-Regular.ttf"
        }
    ]
    
    fonts_dir = Path("fonts")
    
    for font_info in backup_fonts:
        try:
            font_path = fonts_dir / font_info["name"]
            
            print(f"Trying {font_info['name']}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(font_info["url"], headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(font_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded: {font_info['name']}")
            
            # Test this font
            if test_font(font_path):
                return font_path
                
        except Exception as e:
            print(f"✗ {font_info['name']} failed: {e}")
            continue
    
    print("✗ All backup methods failed")
    return None

def test_font(font_path):
    """
    Test a font file with Greek text.
    """
    print(f"Testing font: {font_path.name}")
    
    greek_text = "Αρχεία Βάλτε Χ πριν από την κατάλ"
    
    try:
        # Load the font
        with open(font_path, "rb") as font_file:
            fontbuffer = font_file.read()
        
        font = fitz.Font(fontbuffer=fontbuffer)
        
        # Create test PDF
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        # Test with TextWriter
        tw = fitz.TextWriter(page.rect)
        
        tw.append(
            (50, 100),
            f"Font: {font_path.name}",
            font=font,
            fontsize=14,
            color=(0, 0, 0)
        )
        
        tw.append(
            (50, 140),
            f"Greek: {greek_text}",
            font=font,
            fontsize=12,
            color=(0, 0, 1)
        )
        
        tw.append(
            (50, 180),
            "English: The quick brown fox jumps over the lazy dog",
            font=font,
            fontsize=10,
            color=(0.5, 0.5, 0.5)
        )
        
        tw.write_text(page)
        
        # Save test PDF
        test_pdf = f"test_{font_path.stem}.pdf"
        doc.save(test_pdf)
        doc.close()
        
        print(f"✓ Font test successful! PDF: {test_pdf}")
        return font_path
        
    except Exception as e:
        print(f"✗ Font test failed: {e}")
        return None

def create_simple_font_loader():
    """
    Create a simple font loader for your project.
    """
    font_loader_code = '''"""
Simple Greek font loader.
"""
import fitz
from pathlib import Path

# Global font cache
_font_cache = None

def get_greek_font():
    """Get the Greek-supporting font (cached)."""
    global _font_cache
    
    if _font_cache is None:
        fonts_dir = Path(__file__).parent / "fonts"
        
        # Look for any font file
        font_files = list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf"))
        
        if not font_files:
            raise FileNotFoundError("No font files found in fonts/ directory")
        
        # Use first available font
        font_path = font_files[0]
        
        with open(font_path, "rb") as f:
            fontbuffer = f.read()
        
        _font_cache = fitz.Font(fontbuffer=fontbuffer)
        print(f"Loaded font: {font_path.name}")
    
    return _font_cache

def add_text(page, x, y, text, size=12, color=(0, 0, 0)):
    """Add text to PDF page using Greek font."""
    font = get_greek_font()
    
    tw = fitz.TextWriter(page.rect)
    tw.append((x, y), text, font=font, fontsize=size, color=color)
    tw.write_text(page)

# Test function
if __name__ == "__main__":
    try:
        font = get_greek_font()
        print("✓ Font loaded successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
'''
    
    with open("greek_font.py", "w", encoding="utf-8") as f:
        f.write(font_loader_code)
    
    print("✓ Created greek_font.py")

def main():
    """
    Simple main function.
    """
    print("=== Greek Font Setup ===")
    
    # Download font
    font_path = download_working_font()
    
    if font_path:
        print(f"\n✓ Success! Font ready: {font_path}")
        
        # Create helper module
        create_simple_font_loader()
        
        print("\n=== Usage ===")
        print("In your code:")
        print("""
from greek_font import add_text

# Add Greek text to any PDF page
add_text(page, 50, 100, "Αρχεία Βάλτε Χ πριν από την κατάλ")
""")
        
    else:
        print("\n✗ Font setup failed")
        print("Manual solution:")
        print("1. Go to https://fonts.google.com/noto/specimen/Noto+Sans")
        print("2. Click 'Download family'")
        print("3. Extract NotoSans-Regular.ttf to fonts/ directory")

if __name__ == "__main__":
    main()