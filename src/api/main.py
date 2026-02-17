from fastapi import FastAPI
from src.api.routes import attendance, persons
from src.database.db import engine
from src.database.models import Base

# Create tables 
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Recognition Attendance API",
    version="1.0.0"
)

# Register routers
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])
app.include_router(persons.router, prefix="/persons", tags=["Persons"])

@app.get("/")
def root():
    return {"message": "Attendance API is running"}
