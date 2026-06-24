from sqlalchemy.engine import default
from sqlalchemy import null
from sqlalchemy import Column, String, Integer, DateTime 
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime 
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id  = Column(String, primary_key=True,
                        default= lambda:str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filetype = Column(String, nullable=False)
    status = Column(String, default="processing")
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    