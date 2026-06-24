from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import init_db
from backend.routes.upload import router as upload_router

app = FastAPI(title="Athena API")

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    init_db()

app.include_router(upload_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Athena is running"}