from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
import fitz

app = FastAPI(title="PDF Template Field Finder API", version="1.0.0")

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
    key: str
    label: str
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

class FieldInfo(BaseModel):
    label: str
    description: str

class RequiredFieldsInput(BaseModel):
    required_fields: Dict[str, FieldInfo]

class FieldFinderResponse(BaseModel):
    required_fields: Dict[str, FieldInfo]
    fillable_text_info: List[TextMatch]
    missing_keys: List[str]

# Helper functions
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

def find_template_fields_in_pdf(pdf_bytes: bytes, required_fields: Dict[str, FieldInfo]) -> List[TextMatch]:
    """Search for specific template keys in PDF and return detailed information about matches"""
    matches = []
    
    # Create a set of keys to search for
    target_keys = set(required_fields.keys())
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Create pattern to match any of the target keys
        escaped_keys = [re.escape(key) for key in target_keys]
        pattern = re.compile('|'.join(escaped_keys))
        
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
                        
                        # Search for target key matches in this span
                        for match in pattern.finditer(span_text):
                            matched_key = match.group()
                            
                            # Skip if this key is not in our target list
                            if matched_key not in target_keys:
                                continue
                                
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
                            
                            # Extract context from the full page text
                            page_match_start = page_text.find(matched_key)
                            if page_match_start != -1:
                                context_before, context_after = extract_context(
                                    page_text, page_match_start, page_match_start + len(matched_key)
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
                            
                            # Get the label from required_fields
                            field_info = required_fields[matched_key]
                            
                            # Create match object
                            text_match = TextMatch(
                                key=matched_key,
                                label=field_info.label,
                                value="",  # Empty value as this is a template
                                position=position,
                                font=font_info,
                                alignment=get_alignment(span_bbox, line_bbox),
                                page_number=page_num + 1,
                                block_number=block_num,
                                line_number=line_num,
                                span_number=span_num,
                                rotation=page.rotation,
                                confidence=1.0,
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

# Main extraction endpoint
@app.post("/find-template-fields", response_model=FieldFinderResponse)
async def find_template_fields(
    required_fields_input: RequiredFieldsInput,
    pdf_file: UploadFile = File(..., description="PDF file to search for template keys")
):
    """
    Find specific template keys in PDF based on provided required_fields.
    Only searches for the keys provided in required_fields and returns their positions and details.
    """
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Validate that required_fields is not empty
    if not required_fields_input.required_fields:
        raise HTTPException(status_code=400, detail="required_fields cannot be empty")
    
    try:
        # Read PDF file
        pdf_bytes = await pdf_file.read()
        
        # Find template fields in PDF
        matches = find_template_fields_in_pdf(pdf_bytes, required_fields_input.required_fields)
        
        # Determine which keys were found and which are missing
        found_keys = set(match.key for match in matches)
        all_required_keys = set(required_fields_input.required_fields.keys())
        missing_keys = list(all_required_keys - found_keys)
        
        return FieldFinderResponse(
            required_fields=required_fields_input.required_fields,
            fillable_text_info=matches,
            missing_keys=missing_keys
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Alternative endpoint that accepts required_fields as JSON in request body along with file
@app.post("/find-fields-json")
async def find_fields_with_json_body(
    pdf_file: UploadFile = File(..., description="PDF file to search for template keys"),
    required_fields: str = None
):
    """
    Alternative endpoint where required_fields can be passed as JSON string.
    This is useful when you need to send both file and JSON data in a single request.
    
    Example required_fields JSON string:
    {
        "{first_name}": {"label": "First Name", "description": "Person's first name"},
        "{last_name}": {"label": "Last Name", "description": "Person's last name"}
    }
    """
    import json
    
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Parse required_fields JSON
    if not required_fields:
        raise HTTPException(status_code=400, detail="required_fields JSON string is required")
    
    try:
        required_fields_dict = json.loads(required_fields)
        # Convert to FieldInfo objects
        required_fields_converted = {
            key: FieldInfo(label=value["label"], description=value["description"])
            for key, value in required_fields_dict.items()
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for required_fields")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field in JSON: {e}")
    
    try:
        # Read PDF file
        pdf_bytes = await pdf_file.read()
        
        # Find template fields in PDF
        matches = find_template_fields_in_pdf(pdf_bytes, required_fields_converted)
        
        # Determine which keys were found and which are missing
        found_keys = set(match.key for match in matches)
        all_required_keys = set(required_fields_converted.keys())
        missing_keys = list(all_required_keys - found_keys)
        
        return FieldFinderResponse(
            required_fields=required_fields_converted,
            fillable_text_info=matches,
            missing_keys=missing_keys
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Template Field Finder API",
        "version": "1.0.0",
        "description": "Find specific template keys in PDF files based on provided field definitions",
        "endpoints": {
            "/find-template-fields": "POST - Find template fields in PDF using provided required_fields",
            "/find-fields-json": "POST - Alternative endpoint with JSON string for required_fields",
            "/docs": "GET - API documentation"
        },
        "usage_example": {
            "required_fields": {
                "{first_name}": {
                    "label": "First Name",
                    "description": "Person's first name"
                },
                "{last_name}": {
                    "label": "Last Name", 
                    "description": "Person's last name"
                }
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pdf-template-field-finder"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)