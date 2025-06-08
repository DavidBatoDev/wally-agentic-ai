from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
import json
import fitz
import base64
import google.generativeai as genai
import os

# Configure Gemini AI
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel('gemini-2.0-flash')
    print("Gemini AI configured successfully")
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    model = None

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
    key: str
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

class TemplateExtractionResponse(BaseModel):
    required_fields: Dict[str, str]
    fillable_text_info: List[TextMatch]

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

def search_template_keys_in_pdf(pdf_bytes: bytes) -> List[TextMatch]:
    """Search for template keys in format {key} in PDF and return detailed information about matches"""
    matches = []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Pattern to match {any_text_here} format
        pattern = re.compile(r'\{[^}]+\}')
        
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
                        
                        # Search for template key matches in this span
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
                            
                            # Extract context from the full page text
                            page_match_start = page_text.find(match.group())
                            if page_match_start != -1:
                                context_before, context_after = extract_context(
                                    page_text, page_match_start, page_match_start + len(match.group())
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
                                key=match.group(),
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

async def generate_field_descriptions_with_image(keys: List[str], pdf_content: str, matches: List[TextMatch], pdf_bytes: bytes) -> Dict[str, str]:
    """Generate field descriptions using Gemini AI with PDF context AND visual analysis"""
    if not model:
        # Fallback descriptions if Gemini is not available
        return create_enhanced_fallback_descriptions(keys)
    
    try:
        # Convert PDF pages to images for visual analysis
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_images = []
        
        for page_num in range(min(3, doc.page_count)):  # Analyze first 3 pages max
            page = doc[page_num]
            # Convert page to image (300 DPI for good quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
            img_data = pix.tobytes("png")
            
            # Convert to base64 for Gemini
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            page_images.append({
                "page_number": page_num + 1,
                "image_data": img_base64
            })
        
        doc.close()
        
        # Create enhanced context information from matches
        context_info = []
        for match in matches:
            context_info.append({
                "key": match.key,
                "context_before": match.context_before[:100],
                "context_after": match.context_after[:100],
                "page": match.page_number,
                "position": {
                    "x": match.bbox_center["x"],
                    "y": match.bbox_center["y"]
                },
                "font_size": match.font.size
            })
        
        # Truncate PDF content if too long
        max_content_length = 6000  # Reduced to make room for images
        truncated_content = pdf_content[:max_content_length] if len(pdf_content) > max_content_length else pdf_content
        
        keys_text = ", ".join(keys)
        context_text = json.dumps(context_info, indent=2)
        
        # Enhanced prompt with image analysis instructions and birth certificate expertise
        prompt = f"""
        I have a BIRTH CERTIFICATE PDF form with template keys that need descriptions. I'm providing both the text content AND images of the form pages for comprehensive analysis.

        TEMPLATE KEYS TO DESCRIBE: {keys_text}

        PDF TEXT CONTENT:
        {truncated_content}

        FIELD CONTEXT AND POSITIONS:
        {context_text}

        CRITICAL BIRTH CERTIFICATE FORM KNOWLEDGE:
        This appears to be a Philippines birth certificate form. Here are key sections and their meanings:

        ATTENDANT SECTION:
        - at1, at2, at3, at4, at5 = These are checkboxes for "ATTENDANT" types:
          * at1 = Physician checkbox
          * at2 = Traditional midwife checkbox  
          * at3 = Hilot checkbox
          * at4 = Others checkbox
          * at5 = Others specify field

        MULTIPLE BIRTH SECTION:  
        - imb1, imb2, imb3 = "IF MULTIPLE BIRTH, CHILD WAS" options:
          * imb1 = "1 First" checkbox
          * imb2 = "2 Twin" checkbox  
          * imb3 = "3 Triplet, etc." checkbox
        - imb_o = "Others, Specify" field for multiple birth

        BIRTH ORDER/TYPE SECTION:
        - tb1, tb2, tb3 = Related to "Total number of children born alive"
        - bo = "Birth Order" (first, second, third, etc.)
        - wab = "Weight at Birth"

        GENDER/SEX MARKERS:
        - fe = "Female" checkbox marker
        - ma = "Male" checkbox marker

        CHILDREN COUNT FIELDS:
        - tncba = "Total Number of Children Born Alive"
        - ncslib = "No. of Children Still Living Including this Birth"  
        - ncbobnd = "No. of Children Born alive but are Now Dead"
        - ftb = "Father's Total no. of children Born"
        - mtb = "Mother's Total no. of children Born"

        INSTRUCTIONS:
        1. Analyze the visual layout in the images to confirm field positions
        2. Use birth certificate form knowledge above to provide accurate descriptions
        3. Look for checkbox patterns, text input areas, and form sections
        4. Be specific about what should be marked (X, checkmark, etc.)
        5. Identify the exact purpose based on Philippines birth certificate standards

        Return ONLY a JSON object with precise descriptions:

        {{
            "{{key1}}": "Specific description based on birth certificate form structure",
            "{{key2}}": "Another specific description"
        }}

        Examples of correct descriptions:
        - "Checkbox to mark with X if attendant was a Physician"
        - "Checkbox to mark with X if multiple birth and child was first born (twins/triplets)"
        - "Text input field for child's weight at birth in grams"
        - "Checkbox to mark with X if child is female"
        - "Number input field for total count of children born alive to mother"

        Focus on the EXACT purpose based on Philippines birth certificate form structure.
        """
        
        # Create the content parts for Gemini (text + images)
        content_parts = [prompt]
        
        # Add images to the content
        for page_img in page_images:
            content_parts.append({
                "mime_type": "image/png",
                "data": page_img["image_data"]
            })
        
        # Generate content with both text and images
        response = model.generate_content(content_parts)
        
        # Clean up the response text
        response_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        # Parse the JSON response
        try:
            descriptions = json.loads(response_text)
            # Ensure all keys are present
            result = {}
            for key in keys:
                if key in descriptions:
                    result[key] = descriptions[key]
                else:
                    # Enhanced fallback for missing keys
                    clean_key = key.strip('{}').replace('_', ' ').title()
                    result[key] = f"Form field for {clean_key} (requires visual analysis for precise description)"
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            # Enhanced fallback descriptions
            return create_enhanced_fallback_descriptions(keys)
            
    except Exception as e:
        print(f"Error generating descriptions with Gemini: {e}")
        # Enhanced fallback descriptions
        return create_enhanced_fallback_descriptions(keys)

def create_enhanced_fallback_descriptions(keys: List[str]) -> Dict[str, str]:
    """Create accurate fallback descriptions based on Philippines birth certificate form structure"""
    descriptions = {}
    
    # Birth certificate specific field mappings
    birth_cert_fields = {
        # Attendant checkboxes
        '{at1}': "Checkbox to mark with X if attendant was a Physician",
        '{at2}': "Checkbox to mark with X if attendant was a Traditional Midwife", 
        '{at3}': "Checkbox to mark with X if attendant was a Hilot",
        '{at4}': "Checkbox to mark with X if attendant was Others",
        '{at5}': "Text input field to specify other type of attendant",
        
        # Multiple birth options
        '{imb1}': "Checkbox to mark with X if multiple birth and child was first born",
        '{imb2}': "Checkbox to mark with X if multiple birth and child was twin (second)",
        '{imb3}': "Checkbox to mark with X if multiple birth and child was triplet or more",
        '{imb_o}': "Text input field to specify other multiple birth details",
        
        # Gender markers
        '{fe}': "Checkbox to mark with X if child is female",
        '{ma}': "Checkbox to mark with X if child is male",
        
        # Birth statistics
        '{tb1}': "Number input field for total number of children born alive to mother",
        '{tb2}': "Number input field for number of children still living including this birth", 
        '{tb3}': "Number input field for number of children born alive but now dead",
        '{tncba}': "Number input field for total number of children born alive",
        '{ncslib}': "Number input field for number of children still living including this birth",
        '{ncbobnd}': "Number input field for number of children born alive but now dead",
        '{ftb}': "Number input field for father's total number of children born",
        '{mtb}': "Number input field for mother's total number of children born",
        
        # Birth details
        '{wab}': "Number input field for child's weight at birth in grams",
        '{bo}': "Text input field for child's birth order (first, second, third, etc.)",
        '{cob_t}': "Number input field for total number of children born alive"
    }
    
    for key in keys:
        if key in birth_cert_fields:
            descriptions[key] = birth_cert_fields[key]
        else:
            clean_key = key.strip('{}').lower()
            
            # Standard field patterns
            if 'day' in clean_key:
                descriptions[key] = "Day component of a date (1-31)"
            elif 'month' in clean_key:
                descriptions[key] = "Month component of a date (1-12 or January-December)"
            elif 'year' in clean_key:
                descriptions[key] = "Year component of a date (YYYY format)"
            elif 'first_name' in clean_key:
                descriptions[key] = "Text input field for first name"
            elif 'middle_name' in clean_key:
                descriptions[key] = "Text input field for middle name"
            elif 'last_name' in clean_key:
                descriptions[key] = "Text input field for last name"
            elif 'citizenship' in clean_key:
                descriptions[key] = "Text input field for citizenship information"
            elif 'occupation' in clean_key:
                descriptions[key] = "Text input field for occupation"
            elif 'religion' in clean_key:
                descriptions[key] = "Text input field for religion"
            elif 'address' in clean_key:
                descriptions[key] = "Text input field for address information"
            elif 'province' in clean_key or 'prov' in clean_key:
                descriptions[key] = "Text input field for province"
            elif 'city' in clean_key or 'municipality' in clean_key:
                descriptions[key] = "Text input field for city or municipality"
            elif 'registration' in clean_key:
                descriptions[key] = "Text input field for registration number"
            elif 'date' in clean_key:
                descriptions[key] = "Date input field"
            elif 'place' in clean_key:
                descriptions[key] = "Text input field for location or place"
            elif 'informant' in clean_key:
                descriptions[key] = "Text input field for informant information"
            elif 'prepared' in clean_key:
                descriptions[key] = "Text input field for certificate preparation information"
            elif 'receive' in clean_key:
                descriptions[key] = "Text input field for certificate receipt information"
            elif 'cob' in clean_key:
                descriptions[key] = "Text input field related to birth certification"
            else:
                descriptions[key] = f"Text input field for {clean_key.replace('_', ' ').title()}"
    
    return descriptions

# Main extraction endpoint
@app.post("/extract-generate-template", response_model=TemplateExtractionResponse)
async def extract_text_from_pdf_enhanced(
    pdf_file: UploadFile = File(..., description="PDF file to search for template keys")
):
    """
    Extract template keys in {key} format from PDF and generate field descriptions using AI with PDF context AND visual analysis.
    """
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF file
        pdf_bytes = await pdf_file.read()
        
        # Extract full PDF text content for context
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_pdf_text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            full_pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        
        # Search for template keys
        matches = search_template_keys_in_pdf(pdf_bytes)
        
        # Extract unique keys for description generation
        unique_keys = list(set([match.key for match in matches]))
        
        # Generate field descriptions using enhanced method with image analysis
        required_fields = await generate_field_descriptions_with_image(
            unique_keys, full_pdf_text, matches, pdf_bytes
        )
        
        return TemplateExtractionResponse(
            required_fields=required_fields,
            fillable_text_info=matches
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Text Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract-text-position": "POST - Extract template keys from PDF and generate field descriptions",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pdf-text-extraction"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)