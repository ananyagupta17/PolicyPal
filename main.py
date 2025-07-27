from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.api.routes import router as upload_router

app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update to specific domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload_router)
