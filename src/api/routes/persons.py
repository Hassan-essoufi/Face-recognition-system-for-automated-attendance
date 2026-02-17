from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import os
import uuid
import shutil
import cv2

from src.api.schemas import PersonAdd, PersonRead
from src.api.dependencies import get_db
from src.database.models import Person
from src.recognition.embeddings import build_database
from src.recognition.add_person_db import capture_faces, augment_image

router = APIRouter()

@router.post("/add", response_model=PersonRead)
def add_person(
    payload: PersonAdd = Depends(),
    image_file: UploadFile = File(None), 
    db: Session = Depends(get_db)
):
    """
    Registers a new person:
    - Uses existing recognition logic
    - Updates dataset & embeddings
    - Stores metadata in DB
    """
    # Check duplicate
    existing_person = db.query(Person).filter(Person.name == payload.name).first()
    if existing_person:
        raise HTTPException(status_code=400, detail="Person already exists")
    capture_method = payload.capture_method

    try:
        if capture_method == "webcam":
            capture_faces(
                person_name=payload.name,
                output_dir=payload.output_dir
            )
        elif capture_method == "upload":
            if not image_file:
                raise HTTPException(status_code=400, detail="Image file required for upload method")
            os.makedirs('uploads', exist_ok=True)
            
            unique_filename = f"{uuid.uuid4()}_{image_file.filename}"
            image_path = os.path.join('uploads', unique_filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            
            # Apply augmentation
            augment_image(
                image_path=image_path,
                person_name=payload.name,
                output_dir=payload.output_dir
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid capture method")
            
        build_database()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recognition error: {str(e)}")
    # Person metadata
    person = Person(name=payload.name)
    db.add(person)
    db.commit()
    db.refresh(person)

    return person

@router.get("/", response_model=list[PersonRead])
def list_persons(db: Session = Depends(get_db)):
    return db.query(Person).all()