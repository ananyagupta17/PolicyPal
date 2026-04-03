from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="PolicyPal",
    description="Chat with your policy documents using RAG — Retrieval Augmented Generation.",
    version="2.0.0"
)

# CORS — allows the frontend (different port) to talk to this API
# In production you'd replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api", tags=["PolicyPal"])

# Serve the frontend as static files at "/"
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")