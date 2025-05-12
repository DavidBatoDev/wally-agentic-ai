from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import tempfile
import os
import fitz  # PyMuPDF
from typing import List, Optional
from PIL import Image

app = FastAPI(title="Document-Scanner API",
              description="Warp-&-threshold endpoint suitable for OCR",
              version="1.2.0")

# ---------- geometry helpers ----------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points in (TL, TR, BR, BL) order."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def scan_and_gray(img: np.ndarray) -> np.ndarray:
    """Detect page, warp to bird's-eye view, then just output grayscale."""
    H_img, W_img = img.shape[:2]
    img_area     = H_img * W_img

    # ---------- 1. quick pre-processing ----------
    k = 900 / H_img
    small  = cv2.resize(img, (int(W_img * k), 900))
    gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edge   = cv2.Canny(blur, 60, 180)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge   = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---------- 2. find outer page contour ----------
    cnts,_ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = sorted(cnts, key=cv2.contourArea, reverse=True)

    page = None
    for c in cnts[:10]:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area   = cv2.contourArea(c) / (k * k)
        if len(approx) == 4 and area > 0.20 * img_area:
            page = approx
            break

    if page is None:
        page = np.array([[[0, 0]], [[W_img - 1, 0]],
                         [[W_img - 1, H_img - 1]], [[0, H_img - 1]]],
                        dtype=np.int32)

    # ---------- 3. perspective warp ----------
    rect        = order_points(page)
    (tl, tr, br, bl) = rect
    W           = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H           = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst         = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]],
                           dtype="float32")
    M           = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    warp        = cv2.warpPerspective(img, M, (W, H))

    # ---------- 4. convert to grayscale and return ----------
    warp_gray   = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return warp_gray


def process_file_content(content: bytes, file_ext: str) -> List[np.ndarray]:
    """Process file content based on file extension and return processed images."""
    processed_images = []
    
    if file_ext.lower() == ".pdf":
        # Process PDF file using PyMuPDF (fitz) instead of pdf2image
        try:
            # Load the PDF from bytes
            with fitz.open(stream=content, filetype="pdf") as pdf_document:
                # Process each page
                for page_num in range(len(pdf_document)):
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Render page to an image (PyMuPDF uses Pillow)
                    pix = page.get_pixmap(dpi=300)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Convert PIL Image to OpenCV format
                    open_cv_image = np.array(img)
                    # Convert RGB to BGR (OpenCV format) if needed
                    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
                        open_cv_image = open_cv_image[:, :, ::-1].copy()
                    
                    # Process the image
                    processed = scan_and_gray(open_cv_image)
                    processed_images.append(processed)
        except Exception as e:
            raise ValueError(f"PDF processing error: {str(e)}")
    else:
        # Process image file
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot decode image")
        
        processed = scan_and_gray(img)
        processed_images.append(processed)
    
    return processed_images


# ---------- FastAPI route ----------
@app.post("/scan", summary="Warp & threshold a document photo or PDF")
async def scan(file: UploadFile = File(...), page: Optional[int] = None):
    """
    Upload an image (JPEG/PNG) or PDF file.
    Returns a flattened, high-contrast PNG ready for OCR.
    
    For PDF files:
    - If no page parameter is provided, returns the first page.
    - Provide a specific page number (1-based index) to get that page.
    """
    content = await file.read()
    try:
        # Get file extension from filename
        _, file_ext = os.path.splitext(file.filename)
        
        # Process the file content
        processed_images = process_file_content(content, file_ext)
        
        if not processed_images:
            raise ValueError("No images were processed")
        
        # Handle page selection for PDFs
        if file_ext.lower() == ".pdf":
            if page is not None and 1 <= page <= len(processed_images):
                selected_image = processed_images[page-1]  # Convert to 0-based index
            else:
                selected_image = processed_images[0]  # Default to first page
        else:
            selected_image = processed_images[0]  # For images, there's only one
        
        # Encode the processed image as PNG
        ok, png = cv2.imencode(".png", selected_image)
        if not ok:
            raise ValueError("PNG encoding failed")
        
        return StreamingResponse(io.BytesIO(png.tobytes()),
                                media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/scan/all", summary="Process all pages of a PDF document")
async def scan_all_pages(file: UploadFile = File(...)):
    """
    Upload a PDF file and process all pages.
    Returns a multipart response with all processed pages as PNGs.
    
    For non-PDF files, returns a single processed image.
    """
    content = await file.read()
    try:
        # Get file extension from filename
        _, file_ext = os.path.splitext(file.filename)
        
        # Process the file content
        processed_images = process_file_content(content, file_ext)
        
        if not processed_images:
            raise ValueError("No images were processed")
        
        # For multi-page PDFs, create a ZIP file with all processed pages
        if file_ext.lower() == ".pdf" and len(processed_images) > 1:
            buffer = io.BytesIO()
            import zipfile
            with zipfile.ZipFile(buffer, 'w') as zip_file:
                for i, img in enumerate(processed_images):
                    ok, png = cv2.imencode(".png", img)
                    if not ok:
                        raise ValueError(f"PNG encoding failed for page {i+1}")
                    zip_file.writestr(f"page_{i+1}.png", png.tobytes())
            
            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=processed_pages.zip"}
            )
        else:
            # For single images or single-page PDFs
            ok, png = cv2.imencode(".png", processed_images[0])
            if not ok:
                raise ValueError("PNG encoding failed")
            
            return StreamingResponse(io.BytesIO(png.tobytes()),
                                    media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))