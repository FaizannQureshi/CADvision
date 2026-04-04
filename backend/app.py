"""
CADVision Backend - FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CADVision API",
    description="AI-powered CAD drawing comparison tool",
    version="1.0.0"
)

# CORS: comma-separated origins, or set CORS_ORIGINS in .env / deployment (e.g. your frontend URL)
_default_origins = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "CADVision API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)