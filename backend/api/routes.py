"""
API Routes for CADVision
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.comparison import compare_cad_files
import tempfile
import os
from pathlib import Path

router = APIRouter()

@router.post("/compare")
async def compare_drawings(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Compare two CAD drawings (PNG, JPG, or PDF)
    Returns highlighted differences and AI summary
    """
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files
        file1_path = os.path.join(temp_dir, f"input1{Path(file1.filename).suffix}")
        file2_path = os.path.join(temp_dir, f"input2{Path(file2.filename).suffix}")
        
        with open(file1_path, "wb") as f:
            content = await file1.read()
            f.write(content)
        
        with open(file2_path, "wb") as f:
            content = await file2.read()
            f.write(content)
        
        # Run comparison
        result = await compare_cad_files(file1_path, file2_path, temp_dir)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Comparison failed")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"Error in compare endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup temp directory: {e}")