from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import os
import uuid
import shutil
from threading import Lock
from werkzeug.utils import secure_filename
from backend.src.api.schemas import PersonAdd, PersonRead
from backend.src.api.dependencies import get_db
from backend.src.database.models import Person
from backend.src.recognition.embeddings import build_database
from backend.src.recognition.add_person_db import augment_image, crop_face_from_image

router = APIRouter()
CANCELLED_REQUESTS: set[str] = set()
CANCELLED_REQUESTS_LOCK = Lock()


@router.post("/cancel/{request_id}")
def cancel_add_person_request(request_id: str):
    with CANCELLED_REQUESTS_LOCK:
        CANCELLED_REQUESTS.add(request_id)
    return {"status": "cancelled", "request_id": request_id}

@router.post("/add", response_model=PersonRead)
def add_person(
    payload: PersonAdd = Depends(),
    image_file: UploadFile = File(None),
    request_id: str | None = None,
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

    def is_cancelled() -> bool:
        if not request_id:
            return False
        with CANCELLED_REQUESTS_LOCK:
            return request_id in CANCELLED_REQUESTS

    try:
        if is_cancelled():
            raise HTTPException(status_code=499, detail="Opération annulée")

        if capture_method == "webcam":
            if not image_file:
                raise HTTPException(status_code=400, detail="Captured image file required for webcam method")
            os.makedirs('backend/uploads', exist_ok=True)

            unique_filename = f"webcam_{uuid.uuid4()}_{secure_filename(image_file.filename)}"
            image_path = os.path.join('backend/uploads', unique_filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)

            cropped_image_path = crop_face_from_image(image_path)

            augment_image(
                image_path=cropped_image_path,
                person_name=payload.name,
                should_cancel=is_cancelled
            )
        elif capture_method == "upload":
            if not image_file:
                raise HTTPException(status_code=400, detail="Image file required for upload method")
            os.makedirs('backend/uploads', exist_ok=True)
            
            unique_filename = f"{uuid.uuid4()}_{secure_filename(image_file.filename)}"
            image_path = os.path.join('backend/uploads', unique_filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            
            # Apply augmentation
            augment_image(
                image_path=image_path,
                person_name=payload.name,
                should_cancel=is_cancelled
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid capture method")
            
        if is_cancelled():
            raise HTTPException(status_code=499, detail="Opération annulée")

        build_database()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recognition error: {str(e)}")
    finally:
        if request_id:
            with CANCELLED_REQUESTS_LOCK:
                CANCELLED_REQUESTS.discard(request_id)
    # Person metadata
    person = Person(name=payload.name)
    db.add(person)
    db.commit()
    db.refresh(person)

    return person

@router.get("/", response_model=list[PersonRead])
def list_persons(db: Session = Depends(get_db)):
    return db.query(Person).all()