from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import httpx
import json
import os
from datetime import datetime
import uuid

app = FastAPI(title="Template Management API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration
SUPABASE_URL = "https://ylvmwrvyiamecvnydwvj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlsdm13cnZ5aWFtZWN2bnlkd3ZqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzA5MjkzMSwiZXhwIjoyMDYyNjY4OTMxfQ.6PehkE7I_Q9j8EzzSUC6RGi7Z9QykHcY6Qa20eiLKtM"

# Models
class TemplateCreate(BaseModel):
    doc_type: str
    variation: str
    placeholder_json: Dict[str, Any]

class Template(TemplateCreate):
    id: str
    object_key: str
    created_at: datetime

# Supabase client function
async def get_supabase_client():
    return httpx.AsyncClient(
        base_url=SUPABASE_URL,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        },
    )

@app.post("/templates/", response_model=Template)
async def create_template(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    variation: str = Form(...),
    placeholder_json: str = Form(...),
    client: httpx.AsyncClient = Depends(get_supabase_client)
):
    try:
        # Parse the placeholder_json
        try:
            placeholder_data = json.loads(placeholder_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in placeholder_json")
        
        # Generate unique ID
        template_id = str(uuid.uuid4())
        
        # Create object key for storage
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
        object_key = f"templates/{template_id}{file_extension}"
        
        # Upload file to Supabase Storage
        file_content = await file.read()
        upload_response = await client.post(
            f"/storage/v1/object/templates/{object_key}",
            content=file_content,
            headers={"Content-Type": file.content_type} if file.content_type else {}
        )
        
        if upload_response.status_code != 200:
            raise HTTPException(
                status_code=upload_response.status_code,
                detail=f"Failed to upload file: {upload_response.text}"
            )
        
        # Insert row into templates table
        template_data = {
            "id": template_id,
            "doc_type": doc_type,
            "variation": variation,
            "object_key": object_key,
            "placeholder_json": placeholder_data,
            # created_at will be automatically set by the database
        }
        
        response = await client.post(
            "/rest/v1/templates",
            json=template_data
        )
        
        if response.status_code != 201:
            # If database insert fails, try to delete the uploaded file
            await client.delete(f"/storage/v1/object/templates/{object_key}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create template: {response.text}"
            )
        
        # Get the created record with timestamp
        get_response = await client.get(
            f"/rest/v1/templates?id=eq.{template_id}&select=*"
        )
        
        if get_response.status_code != 200:
            raise HTTPException(
                status_code=get_response.status_code,
                detail=f"Failed to retrieve created template: {get_response.text}"
            )
        
        created_template = get_response.json()[0]
        return created_template
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        await client.aclose()

@app.get("/templates/", response_model=list[Template])
async def list_templates(
    client: httpx.AsyncClient = Depends(get_supabase_client)
):
    try:
        response = await client.get("/rest/v1/templates?select=*")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to retrieve templates: {response.text}"
            )
        
        return response.json()
    finally:
        await client.aclose()

@app.get("/templates/{template_id}", response_model=Template)
async def get_template(
    template_id: str,
    client: httpx.AsyncClient = Depends(get_supabase_client)
):
    try:
        response = await client.get(
            f"/rest/v1/templates?id=eq.{template_id}&select=*"
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to retrieve template: {response.text}"
            )
        
        templates = response.json()
        if not templates:
            raise HTTPException(status_code=404, detail="Template not found")
            
        return templates[0]
    finally:
        await client.aclose()

@app.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    client: httpx.AsyncClient = Depends(get_supabase_client)
):
    try:
        # Get the template to find the object_key
        get_response = await client.get(
            f"/rest/v1/templates?id=eq.{template_id}&select=object_key"
        )
        
        if get_response.status_code != 200:
            raise HTTPException(
                status_code=get_response.status_code,
                detail=f"Failed to retrieve template: {get_response.text}"
            )
        
        templates = get_response.json()
        if not templates:
            raise HTTPException(status_code=404, detail="Template not found")
        
        object_key = templates[0]["object_key"]
        
        # Delete from database
        delete_response = await client.delete(
            f"/rest/v1/templates?id=eq.{template_id}"
        )
        
        if delete_response.status_code not in (200, 204):
            raise HTTPException(
                status_code=delete_response.status_code,
                detail=f"Failed to delete template from database: {delete_response.text}"
            )
        
        # Delete from storage
        storage_delete_response = await client.delete(
            f"/storage/v1/object/templates/{object_key}"
        )
        
        if storage_delete_response.status_code not in (200, 204):
            # Log but don't fail if storage deletion fails
            print(f"Warning: Failed to delete file from storage: {storage_delete_response.text}")
        
        return {"message": "Template deleted successfully"}
    finally:
        await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)