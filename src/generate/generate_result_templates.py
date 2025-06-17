from typing import Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
import fitz
from fitz import Rect, TEXT_ALIGN_CENTER, TEXT_ALIGN_RIGHT, TEXT_ALIGN_JUSTIFY, TEXT_ALIGN_LEFT
from supabase import create_client, Client
import os

# Initialize FastAPI app
app = FastAPI()

try:
    supabase: Client = create_client(
        os.environ.get("SUPABASE_URL", ""),
        os.environ.get("SUPABASE_KEY", "")
    )
    print("Supabase client configured successfully")
except Exception as e:
    print(f"Error configuring Supabase: {e}")
    supabase = None

class FieldMetadata(BaseModel):
    value: Optional[str] = None
    value_status: str = "pending"
    translated_value: Optional[str] = None
    translated_status: str = "pending"

class FillTextEnhancedRequest(BaseModel):
    template_id: str
    template_translated_id: str
    isTranslated: bool
    fields: Dict[str, FieldMetadata]

def get_local_unicode_font():
    """
    Get the path to the local NotoSans font that supports Greek characters.
    Returns the font file path or None if font is not found.
    """
    try:
        # Path to your local NotoSans font
        font_path = os.path.join("fonts", "NotoSans-Regular.ttf")
        
        # Check if font exists
        if os.path.exists(font_path):
            print(f"Found local Unicode font at: {font_path}")
            return font_path
        else:
            print(f"Local font not found at: {font_path}")
            # Try alternative paths
            alternative_paths = [
                "./fonts/NotoSans-Regular.ttf",
                "fonts/NotoSans-Regular.ttf",
                os.path.abspath("fonts/NotoSans-Regular.ttf")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found local Unicode font at alternative path: {alt_path}")
                    return alt_path
            
            print("No local Unicode font found in any expected location")
            return None
            
    except Exception as e:
        print(f"Error accessing local Unicode font: {e}")
        return None

def has_unicode_text(text: str) -> bool:
    """
    Check if text contains Unicode characters that need special font handling.
    """
    return any(ord(char) > 127 for char in text)

def get_safe_font(requested_font: str, has_unicode: bool = False) -> str:
    """
    Map font names to PyMuPDF-compatible font names.
    For Unicode text, we'll use external font files.
    """
    if has_unicode:
        # For Unicode text, we'll return a flag to use external font
        return "unicode-font-needed"
    
    font_mapping = {
        "Helvetica": "helv",
        "Arial": "helv",
        "Times": "times-roman",
        "Times New Roman": "times-roman",
        "Courier": "cour",
        "Courier New": "cour",
    }
    
    # Return mapped font or default to Helvetica
    return font_mapping.get(requested_font, "helv")

@app.post(
    "/insert-text-enhanced",
    summary="Insert text values into PDF using template data with translation support"
)
async def insert_text_enhanced(request: FillTextEnhancedRequest):
    """
    Insert text into PDF using template configuration from Supabase database with translation support.
    Downloads the PDF from the file_url stored in the template record.
    
    Args:
        request: Request containing:
            - template_id: UUID of the original template from Supabase
            - template_translated_id: UUID of the translated template from Supabase
            - isTranslated: Boolean flag to determine which template and values to use
            - fields: Dictionary of field mappings with FieldMetadata objects
    
    Returns:
        Modified PDF file
    """
    
    # 1️⃣ Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    
    # 2️⃣ Determine which template to use based on isTranslated flag
    target_template_id = request.template_translated_id if request.isTranslated else request.template_id
    
    # 3️⃣ Fetch template from Supabase (get both info_json and file_url)
    try:
        response = supabase.table("templates").select("info_json, file_url").eq("id", target_template_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Template with ID {target_template_id} not found")
        
        template_info = response.data["info_json"]
        file_url = response.data["file_url"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
        if not file_url:
            raise HTTPException(status_code=400, detail="Template does not have a file_url")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # 4️⃣ Download PDF from file_url
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            pdf_response = await client.get(file_url)
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF from file_url: {str(e)}")
    
    # 5️⃣ Get local Unicode font for Greek text support
    unicode_font_path = get_local_unicode_font()
    if not unicode_font_path:
        print("Warning: No local Unicode font found. Greek text may not display correctly.")
    
    # 6️⃣ Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Get the field key
        field_key = field.get("key", "")
        
        # Check if we have this field in our request
        if field_key in request.fields:
            field_metadata = request.fields[field_key]
            
            # Determine which value to use based on isTranslated flag
            if request.isTranslated:
                # Use translated_value if available
                if field_metadata.translated_value is not None:
                    updated_field["value"] = field_metadata.translated_value
            else:
                # Use regular value if available
                if field_metadata.value is not None:
                    updated_field["value"] = field_metadata.value
        
        updated_fields.append(updated_field)
    
    # 7️⃣ Open PDF document
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")
    
    # 8️⃣ Insert text for each field that has a value
    try:
        for field in updated_fields:
            # Skip fields without values
            if not field.get("value"):
                continue
                
            page = doc[field["page_number"] - 1]
            position = field["position"]
            bbox = Rect(position["x0"], position["y0"], position["x1"], position["y1"])
            
            # Font handling with robust fallback for Unicode/Greek text
            font_info = field["font"]
            requested_font = font_info.get("name", "Helvetica")
            text_value = field["value"]
            
            # Check if text contains Unicode characters
            has_unicode_chars = has_unicode_text(text_value)
            
            # FORCE TEXT COLOR TO ALWAYS BE BLACK - Override any template color
            r, g, b = 0, 0, 0  # Black color (RGB: 0, 0, 0)
            
            # Alignment handling
            alignment = field.get("alignment", "left").lower()
            font_size = font_info.get("size", 10)
            
            # Try to insert text with proper Unicode support
            text_inserted = False
            
            # Method 1: Try with local NotoSans font if available and text has Unicode chars
            if unicode_font_path and has_unicode_chars:
                try:
                    # Load the local NotoSans font
                    with open(unicode_font_path, "rb") as font_file:
                        fontfile = font_file.read()
                    
                    font = fitz.Font(fontbuffer=fontfile)
                    
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        # Use the font object directly with insert_text
                        text_writer = fitz.TextWriter(page.rect, color=(r, g, b))  # Set color here
                        text_writer.append(
                            (position["x0"], baseline),
                            text_value,
                            font=font,
                            fontsize=font_size
                        )
                        text_writer.write_text(page)
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        # For textbox, we need to use the fontname from the font
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=font.name,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted Unicode text with NotoSans font (black): {text_value[:50]}...")
                    
                except Exception as font_error:
                    print(f"NotoSans font insertion failed: {font_error}")
            
            # Method 2: Try with local font for regular text (even if not Unicode)
            elif unicode_font_path and not has_unicode_chars:
                try:
                    # Use NotoSans for all text for consistency
                    with open(unicode_font_path, "rb") as font_file:
                        fontfile = font_file.read()
                    
                    font = fitz.Font(fontbuffer=fontfile)
                    
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        # Use the font object directly with insert_text
                        text_writer = fitz.TextWriter(page.rect, color=(r, g, b))  # Set color here
                        text_writer.append(
                            (position["x0"], baseline),
                            text_value,
                            font=font,
                            fontsize=font_size
                        )
                        text_writer.write_text(page)
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        # For textbox, we need to use the fontname from the font
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=font.name,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted text with NotoSans font (black): {text_value[:50]}...")
                    
                except Exception as font_error:
                    print(f"NotoSans font insertion failed for regular text: {font_error}")
            
            # Method 3: Fallback to built-in fonts with encoding
            if not text_inserted:
                safe_font = get_safe_font(requested_font, has_unicode_chars)
                if safe_font == "unicode-font-needed":
                    safe_font = "helv"  # Fallback to built-in
                
                try:
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        page.insert_text(
                            (position["x0"], baseline),
                            text_value,
                            fontname=safe_font,
                            fontsize=font_size,
                            color=(r, g, b)  # Always black
                        )
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname=safe_font,
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    text_inserted = True
                    print(f"Successfully inserted text with built-in font (black): {text_value[:50]}...")
                    
                except Exception as builtin_error:
                    print(f"Built-in font insertion failed: {builtin_error}")
            
            # Method 4: Last resort - try without encoding
            if not text_inserted:
                try:
                    if alignment == "left":
                        baseline = position["y1"] - 0.15 * font_size
                        page.insert_text(
                            (position["x0"], baseline),
                            text_value,
                            fontname="helv",
                            fontsize=font_size,
                            color=(r, g, b)  # Always black
                        )
                    else:
                        if alignment == "center":
                            align_flag = TEXT_ALIGN_CENTER
                        elif alignment == "right":
                            align_flag = TEXT_ALIGN_RIGHT
                        elif alignment == "justify":
                            align_flag = TEXT_ALIGN_JUSTIFY
                        else:
                            align_flag = TEXT_ALIGN_LEFT
                        
                        page.insert_textbox(
                            bbox,
                            text_value,
                            fontname="helv",
                            fontsize=font_size,
                            color=(r, g, b),  # Always black
                            align=align_flag
                        )
                    
                    print(f"Inserted text with last resort method (black): {text_value[:50]}...")
                    
                except Exception as final_error:
                    print(f"All text insertion methods failed for: {text_value[:50]}... Error: {final_error}")
        
        # 9️⃣ Save modified PDF to buffer
        buf = io.BytesIO()
        try:
            # Save with garbage collection and compression for better Unicode handling
            doc.save(buf, garbage=4, deflate=True, clean=True)
            print(f"PDF saved successfully with advanced options")
        except Exception as save_error:
            print(f"Error saving with advanced options: {save_error}")
            # Fallback to basic save
            buf = io.BytesIO()  # Reset buffer
            doc.save(buf)
            print(f"PDF saved with basic options")
        
        # Always close the document
        doc.close()
        doc = None  # Clear reference
        
        # Get final buffer size AFTER all operations are complete
        final_buffer_size = buf.tell()
        buf.seek(0)  # Reset position for reading
        
        # Validate PDF content
        if final_buffer_size == 0:
            raise HTTPException(status_code=500, detail="Generated PDF is empty")
            
        # Check if buffer contains PDF header
        pdf_header = buf.read(4)
        buf.seek(0)  # Reset position again
        print(f"PDF header: {pdf_header}")
        
        if not pdf_header.startswith(b'%PDF'):
            raise HTTPException(status_code=500, detail="Generated file is not a valid PDF")
        
    except Exception as e:
        # Make sure to close the document even if there's an error
        if doc:
            doc.close()
        raise HTTPException(status_code=500, detail=f"PDF modification failed: {str(e)}")
    
    # 🔟 Return the modified PDF using StreamingResponse
    template_type = "translated" if request.isTranslated else "original"
    
    # Debug: Log response details
    print(f"Returning PDF: size={final_buffer_size} bytes, template_type={template_type}")
    
    headers = {
        "Content-Disposition": f'attachment; filename="filled_{template_type}_template_{target_template_id}.pdf"',
        "Cache-Control": "no-cache"
    }
    
    print(f"Response headers: {headers}")
    
    return StreamingResponse(
        io.BytesIO(buf.getvalue()),  # Create a fresh BytesIO with the complete data
        media_type="application/pdf", 
        headers=headers
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)