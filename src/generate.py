from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import io
import re
import os
import json
import fitz
from fitz import Rect, TEXT_ALIGN_LEFT, TEXT_ALIGN_CENTER, TEXT_ALIGN_RIGHT, TEXT_ALIGN_JUSTIFY
from fastapi.responses import StreamingResponse

from supabase import create_client, Client

try:
    supabase: Client = create_client(
        os.environ.get("SUPABASE_URL", ""),
        os.environ.get("SUPABASE_KEY", "")
    )
    print("Supabase client configured successfully")
except Exception as e:
    print(f"Error configuring Supabase: {e}")
    supabase = None

app = FastAPI(title="PDF Text Extraction API", version="1.0.0")

# Response models
class FontInfo(BaseModel):
    name: str
    size: float
    style: str
    color: str
    flags: int

class Position(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float

class TextMatch(BaseModel):
    value: str
    position: Position
    font: FontInfo
    alignment: str
    page_number: int
    block_number: int
    line_number: int
    span_number: int
    rotation: float
    confidence: float
    context_before: str
    context_after: str
    bbox_center: Dict[str, float]

class MultiTermMatch(BaseModel):
    search_term: str
    match: TextMatch | None     # None if the term was not found

class MultiExtractionResponse(BaseModel):
    results: List[MultiTermMatch]


class ExtractionResponse(BaseModel):
    matches: List[TextMatch]

class InsertBlock(BaseModel):
    value: str
    position: Position      
    font: FontInfo         
    alignment: str
    page_number: int
    block_number: int
    line_number: int
    span_number: int
    rotation: float
    confidence: float
    context_before: str
    context_after: str
    bbox_center: Dict[str, float]


def get_font_style(flags: int) -> str:
    """Convert font flags to readable style description"""
    styles = []
    if flags & 2**4:  # Bold
        styles.append("bold")
    if flags & 2**1:  # Italic
        styles.append("italic")
    if flags & 2**2:  # Monospace
        styles.append("monospace")
    return " ".join(styles) if styles else "normal"

def get_safe_font(font_name: str) -> str:
    """Return a safe font name, falling back to Helvetica if necessary"""
    # Map common Windows fonts to PDF base-14 fonts
    font_fallback = {
        "ArialMT": "Helvetica",
        "Arial": "Helvetica",
        "Arial-BoldMT": "Helvetica-Bold",
        "TimesNewRomanPSMT": "Times-Roman",
        # Add more mappings as needed
    }
    return font_fallback.get(font_name, font_name) or "Helvetica"

def get_alignment(span_bbox, line_bbox) -> str:
    """Determine text alignment based on position within line"""
    line_width = line_bbox[2] - line_bbox[0]
    span_start = span_bbox[0] - line_bbox[0]
    span_end = line_bbox[2] - span_bbox[2]
    
    if span_start < line_width * 0.1:
        return "left"
    elif span_end < line_width * 0.1:
        return "right"
    elif abs(span_start - span_end) < line_width * 0.1:
        return "center"
    else:
        return "left"

def extract_context(text: str, match_start: int, match_end: int, context_length: int = 50) -> tuple:
    """Extract context before and after the match"""
    start_context = max(0, match_start - context_length)
    end_context = min(len(text), match_end + context_length)
    
    context_before = text[start_context:match_start].strip()
    context_after = text[match_end:end_context].strip()
    
    return context_before, context_after

def search_text_in_pdf(pdf_bytes: bytes, search_term: str, case_sensitive: bool = False) -> List[TextMatch]:
    """Search for text in PDF and return detailed information about matches"""
    matches = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(search_term), flags)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Get text blocks with detailed information
            text_dict = page.get_text("dict")
            page_text = page.get_text()
            
            for block_num, block in enumerate(text_dict["blocks"]):
                if "lines" not in block:
                    continue
                    
                for line_num, line in enumerate(block["lines"]):
                    line_bbox = line["bbox"]
                    
                    for span_num, span in enumerate(line["spans"]):
                        span_text = span["text"]
                        span_bbox = span["bbox"]
                        
                        # Search for matches in this span
                        for match in pattern.finditer(span_text):
                            # Calculate position within the span
                            char_start = match.start()
                            char_end = match.end()
                            
                            # Estimate character positions within bbox
                            chars_in_span = len(span_text)
                            if chars_in_span > 0:
                                char_width = (span_bbox[2] - span_bbox[0]) / chars_in_span
                                
                                # Calculate approximate bbox for the matched text
                                match_x0 = span_bbox[0] + (char_start * char_width)
                                match_x1 = span_bbox[0] + (char_end * char_width)
                                match_y0 = span_bbox[1]
                                match_y1 = span_bbox[3]
                            else:
                                match_x0, match_y0, match_x1, match_y1 = span_bbox
                            
                            # Extract context
                            full_page_match = pattern.search(page_text)
                            if full_page_match:
                                context_before, context_after = extract_context(
                                    page_text, full_page_match.start(), full_page_match.end()
                                )
                            else:
                                context_before = context_after = ""
                            
                            # Get font information
                            font_info = FontInfo(
                                name=span.get("font", "Unknown"),
                                size=span.get("size", 0),
                                style=get_font_style(span.get("flags", 0)),
                                color=f"#{span.get('color', 0):06x}",
                                flags=span.get("flags", 0)
                            )
                            
                            # Calculate position
                            position = Position(
                                x0=match_x0,
                                y0=match_y0,
                                x1=match_x1,
                                y1=match_y1,
                                width=match_x1 - match_x0,
                                height=match_y1 - match_y0
                            )
                            
                            # Create match object
                            text_match = TextMatch(
                                value=match.group(),
                                position=position,
                                font=font_info,
                                alignment=get_alignment(span_bbox, line_bbox),
                                page_number=page_num + 1,
                                block_number=block_num,
                                line_number=line_num,
                                span_number=span_num,
                                rotation=page.rotation,
                                confidence=1.0,  # PyMuPDF generally has high confidence
                                context_before=context_before,
                                context_after=context_after,
                                bbox_center={
                                    "x": (match_x0 + match_x1) / 2,
                                    "y": (match_y0 + match_y1) / 2
                                }
                            )
                            
                            matches.append(text_match)
        
        doc.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    return matches

@app.post("/extract-text-position", response_model=TextMatch)
async def extract_text_from_pdf(
    search_terms: List[str] = Query(..., description="List of terms to search"),
    case_sensitive: bool = Query(False, description="Case-sensitive search?"),
    pdf_file: UploadFile = File(..., description="PDF file to search in")
):
    """
    Extract text from PDF and search for specific terms with detailed positioning information.
    """
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF file
        pdf_bytes = await pdf_file.read()
        
        flat_matches: List[TextMatch] = []
        for term in search_terms:
            hits = search_text_in_pdf(pdf_bytes, term, case_sensitive)
            if hits:                         # keep the first hit for this term
                flat_matches.append(hits[0])
            return flat_matches
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post(
    "/insert-text",
    summary="Insert text using a single TextMatch-style payload"
)
async def insert_text_payload(
    payload_json: str = Form(..., description="The JSON block (see docs)"),
    pdf_file: UploadFile = File(..., description="PDF to modify")
):
    # ── parse payload ────────────────────────────────────────────────────────
    try:
        block = InsertBlock.model_validate(json.loads(payload_json))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}")

    # sanity-check file
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # ── open PDF ─────────────────────────────────────────────────────────────
    pdf_bytes = await pdf_file.read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")

    # ── compute rectangle & colours ──────────────────────────────────────────
    page_index = block.page_number - 1
    bbox = Rect(block.position.x0, block.position.y0,
                block.position.x1, block.position.y1)

    # Convert colour from hex (#RRGGBB) to 0-1 tuple
    r, g, b = tuple(int(block.font.color.lstrip("#")[i:i+2], 16) / 255
                    for i in (0, 2, 4))

    # ── draw text exactly inside that bbox ───────────────────────────────────
    try:
        page = doc[page_index]
        page.insert_textbox(
            bbox,
            block.value,                     
            fontname=block.font.name or "Helvetica",
            fontsize=block.font.size or 10,
            color=(r, g, b),
            align=TEXT_ALIGN_LEFT
        )

        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF edit failed: {e}")

    headers = {
        "Content-Disposition":
            f'attachment; filename="modified_{pdf_file.filename}"'
    }
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


class BatchInsertBlock(InsertBlock):
    key: str                     # keep the template name (“{province}”, …)

# optional: map common Windows fonts → PDF base-14
FONT_FALLBACK = {
    "ArialMT": "Helvetica",
    "Arial": "Helvetica",
    "Arial-BoldMT": "Helvetica-Bold",
    "TimesNewRomanPSMT": "Times-Roman",
    # add more if you need …
}

class FillableDataRequest(BaseModel):
    template_id: str
    data: Dict[str, str]

# Replace your existing insert-text-batch endpoint with this:
# Add these imports at the top with your existing imports
from supabase import create_client, Client

# Add this new Pydantic model for the request body
class FillableDataRequest(BaseModel):
    template_id: str  # Changed to str to handle UUID
    data: Dict[str, str]

# Replace your existing insert-text-batch endpoint with this:
@app.post(
    "/insert-text-batch",
    summary="Insert text values into PDF using template data from Supabase"
)
async def insert_text_batch(
    pdf_file: UploadFile = File(..., description="PDF to modify"),
    template_id: str = Form(..., description="Template UUID from Supabase"),
    data: str = Form(..., description="JSON string of field values")
):
    """
    Insert text into PDF using template configuration from Supabase database.
    
    Args:
        pdf_file: PDF file to modify
        template_id: UUID of the template from Supabase
        data: JSON string of field mappings (e.g., '{"{province}": "Ontario"}')
    
    Returns:
        Modified PDF file
    """
    
    # 1️⃣ Parse the data JSON string
    try:
        field_data = json.loads(data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in data parameter: {str(e)}")
    
    # 2️⃣ Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    
    # 3️⃣ Fetch template from Supabase
    try:
        response = supabase.table("templates").select("info_json").eq("id", template_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Template with ID {template_id} not found")
        
        template_info = response.data["info_json"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # 4️⃣ Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Update the value if provided in request data
        field_key = field.get("key", "")
        if field_key in field_data:
            updated_field["value"] = field_data[field_key]
        
        updated_fields.append(updated_field)
    
    # 5️⃣ Validate PDF file
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    pdf_bytes = await pdf_file.read()
    
    # 6️⃣ Open PDF document
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")
    
    # 7️⃣ Insert text for each field that has a value
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
                # r = int(color_hex[1:3], 16) / 255
                # g = int(color_hex[3:5], 16) / 255
                # b = int(color_hex[5:7], 16) / 255
                # for now let's use black 
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
        
        # 8️⃣ Save modified PDF to buffer
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF modification failed: {str(e)}")
    
    # 9️⃣ Return the updated fields info and the modified PDF
    # For now, we'll just return the PDF. If you want to return the JSON data instead,
    # you can return updated_fields directly
    
    headers = {
        "Content-Disposition": f'attachment; filename="filled_{pdf_file.filename}"'
    }
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


# Alternative endpoint if you want to return just the populated field data without modifying PDF
@app.post(
    "/get-populated-fields",
    summary="Get populated field data from template without modifying PDF"
)
async def get_populated_fields(
    template_id: str = Form(..., description="Template UUID from Supabase"),
    data: str = Form(..., description="JSON string of field values")
):
    """
    Get populated field information from template without modifying PDF.
    Returns the fillable_text_info with populated values.
    """
    
    # Parse the data JSON string
    try:
        field_data = json.loads(data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in data parameter: {str(e)}")
    
    # Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    
    # Fetch template from Supabase
    try:
        response = supabase.table("templates").select("info_json").eq("id", template_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Template with ID {template_id} not found")
        
        template_info = response.data["info_json"]
        fillable_text_info = template_info.get("fillable_text_info", [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")
    
    # Create updated fillable_text_info with provided values
    updated_fields = []
    
    for field in fillable_text_info:
        # Create a copy of the field
        updated_field = field.copy()
        
        # Update the value if provided in request data
        field_key = field.get("key", "")
        if field_key in field_data:
            updated_field["value"] = field_data[field_key]
        
        updated_fields.append(updated_field)
    
    return updated_fields


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Text Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract-text": "POST - Extract and search text from PDF",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pdf-text-extraction"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)