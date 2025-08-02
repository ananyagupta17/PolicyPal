from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.api.routes import router as pipeline_router
from dotenv import load_dotenv
import os 

load_dotenv()  
print("ENV token loaded:", os.getenv("BEARER_TOKEN"))
app = FastAPI(
    title="Policy QA Pipeline",
    description="Extracts document text and answers user questions using a retrieval-augmented QA system.",
    version="1.0.0"
)

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Updated prefix to match HackRx requirement
app.include_router(pipeline_router, prefix="/api/v1", tags=["Pipeline"])
