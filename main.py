from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
print("ENV token loaded:", os.getenv("BEARER_TOKEN"))

# Logging setup
logging.basicConfig(level=logging.INFO)

# FastAPI instance
app = FastAPI(
    title="Policy QA Pipeline",
    description="Extracts document text, stores embeddings in Pinecone, retrieves relevant chunks, and answers user questions.",
    version="1.0.0"
)

# Enable CORS (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import your main QA routes
from backend.routes.qa_routes import router as qa_router

# Register routes
app.include_router(qa_router, prefix="/api/v1", tags=["QuestionAnswering"])
