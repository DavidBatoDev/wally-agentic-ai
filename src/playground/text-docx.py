from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
import io
import os
import tempfile
from spire.doc import *
from spire.doc.common import *
import httpx
from supabase import create_client, Client

# Initialize Supabase client
try:
    supabase: Client = create_client(
        os.environ.get("SUPABASE_URL", ""),
        os.environ.get("SUPABASE_KEY", "")
    )
    print("Supabase client configured successfully")
except Exception as e:
    print(f"Error configuring Supabase: {e}")
    supabase = None

# Initialize FastAPI app
app = FastAPI(title="Word Template Fill API", version="1.0.0")

# Request model
class FillTextBatchRequest(BaseModel):
    template_id: str
    filled_fields: Dict[str, str]  # e.g., {"{first_name}": "John"}

# Font fallback mapping for Word documents
FONT_FALLBACK = {
    "ArialMT": "Arial",
    "Arial": "Arial", 
    "Arial-BoldMT": "Arial",
    "TimesNewRomanPSMT": "Times New Roman",
    "Helvetica": "Arial",
    "Helvetica-Bold": "Arial",
    # Add more mappings as needed
}

def get_safe_font(font_name: str) -> str:
    """Return a safe font name, falling back to Arial if necessary"""
    return FONT_FALLBACK.get(font_name, font_name) or "Arial"

def hex_to_color(hex_color: str) -> Color:
    """Convert hex color to Spire.Doc Color object"""
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return Color.FromArgb(r, g, b)
    except:
        # Default to black if conversion fails
        return Color.get_Black()

def points_to_pixels(points: float) -> float:
    """Convert points to pixels (approximate conversion)"""
    return points * 1.33  # 1 point ≈ 1.33 pixels at 96 DPI

def get_alignment_type(alignment: str) -> TextAlignment:
    """Convert alignment string to Spire.Doc TextAlignment"""
    alignment_map = {
        "left": TextAlignment.Left,
        "center": TextAlignment.Center,
        "right": TextAlignment.Right,
        "justify": TextAlignment.Justify
    }
    return alignment_map.get(alignment.lower(), TextAlignment.Left)

@app.post(
    "/insert-text-batch",
    summary="Insert text values into Word document using template data from Supabase"
)
async def insert_text_batch(request: FillTextBatchRequest):
    """
    Insert text into Word document using template configuration from Supabase database.
    Downloads the Word document from the file_url stored in the template record.
    
    Args:
        request: Request containing template_id and filled_fields
            - template_id: UUID of the template from Supabase
            - filled_fields: Dictionary of field mappings (e.g., {"{first_name}": "John"})
    
    Returns:
        Modified Word document file
    """
    
    # 1️⃣ Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    
    # 2️⃣ Fetch template from Supabase (get both info_json and file_url)
    try:
        response = supabase.table("templates").select("info_json, file_url").eq("id", request.template_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Template with ID {request.template_id} not found")
        
        template_info = response.data["info_json"]
        file_url = response.data["file_url"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
        if not file_url:
            raise HTTPException(status_code=400, detail="Template does not have a file_url")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # 3️⃣ Download Word document from file_url
    try:
        async with httpx.AsyncClient() as client:
            doc_response = await client.get(file_url)
            doc_response.raise_for_status()
            doc_bytes = doc_response.content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading Word document from file_url: {str(e)}")
    
    # 4️⃣ Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Update the value if provided in request filled_fields
        field_key = field.get("key", "")
        if field_key in request.filled_fields:
            updated_field["value"] = request.filled_fields[field_key]
        
        updated_fields.append(updated_field)
    
    # 5️⃣ Create temporary file for Word document processing
    temp_input_path = None
    temp_output_path = None
    
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_input:
            temp_input.write(doc_bytes)
            temp_input_path = temp_input.name
        
        # Create temporary output file path
        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='.docx')
        os.close(temp_output_fd)
        
        # Load the Word document
        document = Document()
        document.LoadFromFile(temp_input_path)
        
        # 6️⃣ Insert text for each field that has a value
        for field in updated_fields:
            # Skip fields without values
            if not field.get("value"):
                continue
            
            try:
                # Get page and position information
                page_number = field.get("page_number", 1) - 1  # Convert to 0-based index
                position = field["position"]
                font_info = field["font"]
                
                # Ensure we have enough sections
                while document.Sections.Count <= page_number:
                    document.AddSection()
                
                section = document.Sections[page_number]
                
                # Convert coordinates from points to pixels
                x_pos = points_to_pixels(position["x0"])
                y_pos = points_to_pixels(position["y0"])
                width = points_to_pixels(position.get("width", position["x1"] - position["x0"]))
                height = points_to_pixels(position.get("height", position["y1"] - position["y0"]))
                
                # Create a paragraph and textbox for precise positioning
                paragraph = section.AddParagraph()
                textbox = paragraph.AppendTextBox(width, height)
                
                # Apply positioning (with error handling)
                try:
                    textbox.Format.HorizontalOrigin = HorizontalOrigin.Page
                    textbox.Format.VerticalOrigin = VerticalOrigin.Page
                    textbox.Format.HorizontalPosition = x_pos
                    textbox.Format.VerticalPosition = y_pos
                    textbox.Format.TextWrappingStyle = TextWrappingStyle.InFrontOfText
                    textbox.Format.TextWrappingType = TextWrappingType.Both
                except Exception as pos_error:
                    print(f"Warning: Could not set positioning for field {field.get('key', 'unknown')}: {pos_error}")
                
                # Add text to the textbox
                textbox_paragraph = textbox.Body.AddParagraph()
                textbox_text = textbox_paragraph.AppendText(field["value"])
                
                # Apply formatting to textbox text
                safe_font = get_safe_font(font_info.get("name", "Arial"))
                try:
                    textbox_text.CharacterFormat.FontName = safe_font
                    textbox_text.CharacterFormat.FontSize = font_info.get("size", 10)
                    textbox_text.CharacterFormat.TextColor = hex_to_color(font_info.get("color", "#000000"))
                    textbox_paragraph.Format.HorizontalAlignment = get_alignment_type(field.get("alignment", "left"))
                except Exception as format_error:
                    print(f"Warning: Could not apply formatting for field {field.get('key', 'unknown')}: {format_error}")
                
            except Exception as field_error:
                print(f"Error processing field {field.get('key', 'unknown')}: {field_error}")
                continue
        
        # 7️⃣ Save the modified document
        document.SaveToFile(temp_output_path, FileFormat.Docx2016)
        document.Close()
        
        # Read the modified document
        with open(temp_output_path, 'rb') as f:
            modified_doc_bytes = f.read()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word document modification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
    
    # 8️⃣ Return the modified Word document
    buf = io.BytesIO(modified_doc_bytes)
    headers = {
        "Content-Disposition": f'attachment; filename="filled_template_{request.template_id}.docx"'
    }
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)

@app.post(
    "/get-populated-fields",
    summary="Get populated field data from template without modifying Word document"
)
async def get_populated_fields(request: FillTextBatchRequest):
    """
    Get populated field information from template without modifying Word document.
    Returns the fillable_text_info with populated values.
    
    Args:
        request: Request containing template_id and filled_fields
            - template_id: UUID of the template from Supabase
            - filled_fields: Dictionary of field mappings (e.g., {"{first_name}": "John"})
    
    Returns:
        List of field objects with populated values
    """
    
    # Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    
    # Fetch template from Supabase
    try:
        response = supabase.table("templates").select("info_json").eq("id", request.template_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Template with ID {request.template_id} not found")
        
        template_info = response.data["info_json"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Update the value if provided in request filled_fields
        field_key = field.get("key", "")
        if field_key in request.filled_fields:
            updated_field["value"] = request.filled_fields[field_key]
        
        updated_fields.append(updated_field)
    
    return updated_fields

@app.post(
    "/demo-insert",
    summary="Demo endpoint with predefined coordinates for testing"
)
async def demo_insert_text():
    """
    Demo endpoint that creates a sample Word document with text inserted at predefined coordinates.
    Uses the coordinates you provided for demonstration purposes.
    """
    
    try:
        # Create a new document
        document = Document()
        
        # Add a section if it doesn't exist
        if document.Sections.Count == 0:
            section = document.AddSection()
        else:
            section = document.Sections[0]
        
        # Demo field data using your provided coordinates
        demo_field = {
            "key": "{first_name}",
            "value": "John Doe",
            "font": {
                "name": "ArialMT",
                "size": 8.039999961853027,
                "color": "#ee0000",
                "flags": 0,
                "style": "normal"
            },
            "label": "Child's First Name",
            "position": {
                "x0": 129.6199951171875,
                "x1": 172.38536071777344,
                "y0": 144.3037567138672,
                "y1": 153.2683563232422,
                "width": 42.76536560058594,
                "height": 8.964599609375
            },
            "rotation": 0,
            "alignment": "left",
            "page_number": 1
        }
        
        # Add some content to make the document more visible
        intro_paragraph = section.AddParagraph()
        intro_paragraph.AppendText("Demo Word Document with Positioned Text")
        intro_paragraph.Format.HorizontalAlignment = TextAlignment.Center
        
        # Add some spacing
        for _ in range(5):
            section.AddParagraph()
        
        # Insert text at specified coordinates
        position = demo_field["position"]
        font_info = demo_field["font"]
        
        # Convert coordinates
        x_pos = points_to_pixels(position["x0"])
        y_pos = points_to_pixels(position["y0"])
        width = points_to_pixels(position["width"])
        height = points_to_pixels(position["height"])
        
        # Create paragraph for textbox
        paragraph = section.AddParagraph()
        
        # Create textbox with precise positioning
        textbox = paragraph.AppendTextBox(width, height)
        
        # Set positioning properties
        try:
            textbox.Format.HorizontalOrigin = HorizontalOrigin.Page
            textbox.Format.VerticalOrigin = VerticalOrigin.Page
            textbox.Format.HorizontalPosition = x_pos
            textbox.Format.VerticalPosition = y_pos
            textbox.Format.TextWrappingStyle = TextWrappingStyle.InFrontOfText
            textbox.Format.TextWrappingType = TextWrappingType.Both
        except Exception as pos_error:
            print(f"Warning: Could not set textbox positioning: {pos_error}")
        
        # Add text to textbox
        textbox_paragraph = textbox.Body.AddParagraph()
        textbox_text = textbox_paragraph.AppendText(demo_field["value"])
        
        # Apply formatting
        safe_font = get_safe_font(font_info["name"])
        try:
            textbox_text.CharacterFormat.FontName = safe_font
            textbox_text.CharacterFormat.FontSize = font_info["size"]
            textbox_text.CharacterFormat.TextColor = hex_to_color(font_info["color"])
            textbox_paragraph.Format.HorizontalAlignment = get_alignment_type(demo_field["alignment"])
        except Exception as format_error:
            print(f"Warning: Could not apply formatting: {format_error}")
        
        # Don't remove the original paragraph as it may cause issues
        # section.Body.ChildObjects.Remove(paragraph)
        
        # Save to temporary file and return
        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='.docx')
        os.close(temp_output_fd)
        
        try:
            document.SaveToFile(temp_output_path, FileFormat.Docx2016)
            document.Close()
            
            with open(temp_output_path, 'rb') as f:
                doc_bytes = f.read()
            
        finally:
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
        
        buf = io.BytesIO(doc_bytes)
        headers = {
            "Content-Disposition": 'attachment; filename="demo_positioned_text.docx"'
        }
        return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo document creation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Word Template Fill API with Spire.Doc",
        "version": "1.0.0",
        "endpoints": {
            "/insert-text-batch": "POST - Fill Word template with data and return modified document",
            "/get-populated-fields": "POST - Get populated field data without modifying document",
            "/demo-insert": "POST - Demo endpoint with predefined coordinates",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "word-template-fill"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)