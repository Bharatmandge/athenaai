from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.document import Base, Document

DATABASE_URL = "sqlite:///athena.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ SQLite initialized")


def update_doc_status(doc_id: str, status: str):
    """
    Update document processing status.
    """

    db = SessionLocal()

    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()

        if doc:
            doc.status = status
            db.commit()
            db.refresh(doc)

        return doc

    finally:
        db.close()