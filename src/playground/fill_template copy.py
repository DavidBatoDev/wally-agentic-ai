from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict
import io
import os
import fitz
from fitz import Rect, TEXT_ALIGN_LEFT, TEXT_ALIGN_CENTER, TEXT_ALIGN_RIGHT, TEXT_ALIGN_JUSTIFY
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
app = FastAPI(title="PDF Template Fill API", version="1.0.0")

# Request model
class FillTextBatchRequest(BaseModel):
    template_id: str
    filled_fields: Dict[str, str]  # e.g., {"{province}": "Ontario", "{city}": "Toronto"}

# Font fallback mapping
FONT_FALLBACK = {
    "ArialMT": "Helvetica-Bold",
    "Arial": "Helvetica-Bold", 
    "Arial-BoldMT": "Helvetica-Bold",
    "TimesNewRomanPSMT": "Times-Roman",
    # Add more mappings as needed
}

def get_safe_font(font_name: str) -> str:
    """Return a safe font name, falling back to Helvetica if necessary"""
    return FONT_FALLBACK.get(font_name, font_name) or "Helvetica"

@app.post(
    "/insert-text-batch",
    summary="Insert text values into PDF using template data from Supabase"
)
async def insert_text_batch(request: FillTextBatchRequest):
    """
    Insert text into PDF using template configuration from Supabase database.
    Downloads the PDF from the file_url stored in the template record.
    
    Args:
        request: Request containing template_id and filled_fields
            - template_id: UUID of the template from Supabase
            - filled_fields: Dictionary of field mappings (e.g., {"{province}": "Ontario"})
    
    Returns:
        Modified PDF file
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
    
    # 3️⃣ Download PDF from file_url
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            pdf_response = await client.get(file_url)
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF from file_url: {str(e)}")
    
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
    
    # 5️⃣ Open PDF document
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")
    
    # 6️⃣ Insert text for each field that has a value
    try:
        for field in updated_fields:
            # Skip fields without values
            if not field.get("value"):
                continue
                
            page = doc[field["page_number"] - 1]
            position = field["position"]
            bbox = Rect(position["x0"], position["y0"], position["x1"], position["y1"])
            
            # Font handling with robust fallback
            font_info = field["font"]
            requested_font = font_info.get("name", "Helvetica")
            safe_font = get_safe_font(requested_font)
            
            # Log font fallback for debugging (optional)
            if requested_font != safe_font:
                print(f"Font fallback: '{requested_font}' -> '{safe_font}'")
            
            # Color handling (default to black if not specified)
            color_hex = font_info.get("color", "#000000")
            if color_hex.startswith("#"):
                # For now, use black color (you can implement hex to RGB conversion if needed)
                r, g, b = 0, 0, 0
            else:
                r, g, b = 0, 0, 0
            
            # Alignment handling
            alignment = field.get("alignment", "left").lower()
            
            if alignment == "left":
                # For left alignment, use insert_text with baseline positioning
                baseline = position["y1"] - 0.15 * font_info.get("size", 10)
                try:
                    page.insert_text(
                        (position["x0"], baseline),
                        field["value"],
                        fontname=safe_font,
                        fontsize=font_info.get("size", 10),
                        color=(r, g, b)
                    )
                except Exception as font_error:
                    # Ultimate fallback: use Helvetica
                    print(f"Font error with '{safe_font}', using Helvetica: {font_error}")
                    page.insert_text(
                        (position["x0"], baseline),
                        field["value"],
                        fontname="Helvetica",
                        fontsize=font_info.get("size", 10),
                        color=(r, g, b)
                    )
            else:
                # For other alignments, use insert_textbox
                if alignment == "center":
                    align_flag = TEXT_ALIGN_CENTER
                elif alignment == "right":
                    align_flag = TEXT_ALIGN_RIGHT
                elif alignment == "justify":
                    align_flag = TEXT_ALIGN_JUSTIFY
                else:
                    align_flag = TEXT_ALIGN_LEFT
                
                try:
                    page.insert_textbox(
                        bbox,
                        field["value"],
                        fontname=safe_font,
                        fontsize=font_info.get("size", 10),
                        color=(r, g, b),
                        align=align_flag
                    )
                except Exception as font_error:
                    # Ultimate fallback: use Helvetica
                    print(f"Font error with '{safe_font}', using Helvetica: {font_error}")
                    page.insert_textbox(
                        bbox,
                        field["value"],
                        fontname="Helvetica",
                        fontsize=font_info.get("size", 10),
                        color=(r, g, b),
                        align=align_flag
                    )
        
        # 7️⃣ Save modified PDF to buffer
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF modification failed: {str(e)}")
    
    # 8️⃣ Return the modified PDF
    headers = {
        "Content-Disposition": f'attachment; filename="filled_template_{request.template_id}.pdf"'
    }
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)

@app.post(
    "/get-populated-fields",
    summary="Get populated field data from template without modifying PDF"
)
async def get_populated_fields(request: FillTextBatchRequest):
    """
    Get populated field information from template without modifying PDF.
    Returns the fillable_text_info with populated values.
    
    Args:
        request: Request containing template_id and filled_fields
            - template_id: UUID of the template from Supabase
            - filled_fields: Dictionary of field mappings (e.g., {"{province}": "Ontario"})
    
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Template Fill API",
        "version": "1.0.0",
        "endpoints": {
            "/insert-text-batch": "POST - Fill PDF template with data and return modified PDF",
            "/get-populated-fields": "POST - Get populated field data without modifying PDF",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pdf-template-fill"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)