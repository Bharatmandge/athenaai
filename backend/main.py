from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.services.vector_store import create_collection

from backend.routes.upload import router as upload_router
from backend.routes.search import router as search_router


# Create FastAPI app
app = FastAPI(title="Athena API")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup Event
@app.on_event("startup")
async def startup():
    # Initialize SQLite database
    init_db()

    # Initialize Qdrant collection
    create_collection()


# Register Routers
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(search_router, prefix="/api", tags=["Search"])


# Health Check
@app.get("/")
def home():
    return {
        "message": "Athena API is running 🚀"
    }