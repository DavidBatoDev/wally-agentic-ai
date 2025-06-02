# backend/src/agent/analyze_tools.py
from typing import Dict
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import fitz   # PyMuPDF – swap out for your own library

class AnalyzeUploadInput(BaseModel):
    file_id:     str  = Field(...,  description="Supabase UUID of the uploaded file")
    public_url:  str  = Field(...,  description="Signed or public URL to download the file")

def analyze_upload(file_id: str, public_url: str) -> Dict:
    """
    *Download → inspect first page → return simple metadata.*
    Replace this stub with your OCR / heuristics / GPT-Vision call.
    """
    # naïve example: fetch the file & read the first page size
    import requests, tempfile, os, mimetypes
    tmp_file = tempfile.mktemp()
    with open(tmp_file, "wb") as f:
        f.write(requests.get(public_url).content)

    doc = fitz.open(tmp_file)
    first = doc[0]
    w, h  = first.rect.width, first.rect.height

    # super crude rules:
    if w > h:   doc_type, variation = "passport", "horizontal"
    else:       doc_type, variation = "passport", "vertical"

    mime = mimetypes.guess_type(tmp_file)[0] or "application/pdf"
    os.remove(tmp_file)

    return {
        "file_id": file_id,
        "mime_type": mime,
        "doc_type": doc_type,
        "variation": variation,
        "page_size": f"{w:.0f}×{h:.0f}",
    }

# Export exactly ONE StructuredTool
def get_analyze_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=analyze_upload,
        name="analyze_upload",
        description="Always run this to guess the document type & variation",
        args_schema=AnalyzeUploadInput,
        return_direct=True,          # we want the JSON straight back
    )
