from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import io
import re      
import json
import fitz
from fitz import Rect, TEXT_ALIGN_LEFT, TEXT_ALIGN_CENTER, TEXT_ALIGN_RIGHT, TEXT_ALIGN_JUSTIFY
from fastapi.responses import StreamingResponse

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

@app.post("/extract-text", response_model=TextMatch)
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
            return flat_matches[0]
        
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

@app.post(
    "/insert-text-batch",
    summary="Insert many values into a PDF using an array of field details"
)
async def insert_text_batch(
    payload_json: str = Form(..., description="JSON array with field specs"),
    pdf_file: UploadFile = File(..., description="PDF to modify")
):
    # 1️⃣  Parse / validate payload
    try:
        raw_list = json.loads(payload_json)
        blocks: List[BatchInsertBlock] = [
            BatchInsertBlock.model_validate(item) for item in raw_list
        ]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}")

    # 2️⃣  Make sure we got a PDF
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    pdf_bytes = await pdf_file.read()

    # 3️⃣  Open document
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot open PDF: {e}")

    # 4️⃣  Loop & draw
    try:
        for blk in blocks:
            page = doc[blk.page_number - 1]
            bbox = Rect(blk.position.x0, blk.position.y0,
                        blk.position.x1, blk.position.y1)

            # font handling
            requested_font = blk.font.name or "Helvetica"
            font_name = FONT_FALLBACK.get(requested_font, requested_font)

            # colour 0-1 tuple
            # r, g, b = tuple(int(blk.font.color.lstrip("#")[i:i+2], 16) / 255
            #                 for i in (0, 2, 4))
            
            # black if no colour specified
            r,g,b = 0, 0, 0

            align = blk.alignment.lower()
            if align == "left":
                # baseline anchor so it's not hidden under lines
                baseline = blk.position.y1 - 0.15 * (blk.font.size or 10)
                page.insert_text(
                    (blk.position.x0, baseline),
                    blk.value,
                    fontname=font_name,
                    fontsize=blk.font.size or 10,
                    color=(r, g, b)
                )

            else:
                # choose the TEXT_ALIGN constant
                if align == "center":
                    flag = TEXT_ALIGN_CENTER
                elif align == "right":
                    flag = TEXT_ALIGN_RIGHT
                elif align == "justify":
                    flag = TEXT_ALIGN_JUSTIFY
                else:
                    flag = TEXT_ALIGN_LEFT

                page.insert_textbox(
                    bbox,
                    blk.value,
                    fontname=font_name,
                    fontsize=blk.font.size or 10,
                    color=(r, g, b),
                    align=flag
                )

        # save to buffer
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF edit failed: {e}")

    # 5️⃣  stream back
    headers = {
        "Content-Disposition":
            f'attachment; filename="modified_{pdf_file.filename}"'
    }
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


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