from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import PersonAdd, PersonRead
from src.api.dependencies import get_db
from src.database.models import Person
from src.recognition.add_person_db import capture_faces

router = APIRouter()

@router.post("/add", response_model=PersonRead)
def add_person(
    payload: PersonAdd,
    db: Session = Depends(get_db)
):
    """
    Registers a new person:
    - Uses existing recognition logic
    - Updates dataset & embeddings
    - Stores metadata in DB
    """

    try:
        capture_faces(
            person_name=payload.name,
            output_dir=payload.output_dir
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Recognition error: {str(e)}"
        )
    # Person metadata
    person = Person(name=payload.name)
    db.add(person)
    db.commit()
    db.refresh(person)

    return person

@router.get("/", response_model=list[PersonRead])
def list_persons(db: Session = Depends(get_db)):
    return db.query(Person).all()
