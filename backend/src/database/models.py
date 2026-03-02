from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String, index=True)
    date = Column(String)           # YYYY-MM-DD
    time = Column(String)           # HH:MM:SS
    status = Column(String)         # present / absent
    confidence = Column(Float)
    source = Column(String)         # webcam / video

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now())
