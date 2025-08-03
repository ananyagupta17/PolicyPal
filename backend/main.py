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
    description="Extracts document text and answers user questions using a retrieval-augmented QA system.",
    version="1.0.0"
)

# Enable CORS (configure specific origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from backend.app.routes import router as pipeline_router  # /api/v1/*
from backend.routes.qa_routes import router as qa_router  # /api/v1/qa/*

# Register routes
app.include_router(pipeline_router, prefix="/api/v1", tags=["Pipeline"])
app.include_router(qa_router, prefix="/api/v1/qa", tags=["QuestionAnswering"])
