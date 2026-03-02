from sqlalchemy.orm import Session
from src.database.models import Attendance, Person
from datetime import datetime

# Attendance CRUD
def add_attendance(db,data):
    record = Attendance(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_all_attendance(db):
    return db.query(Attendance).all()

def get_today_attendance(db):
    today = datetime.now().strftime("%Y-%m-%d")
    return db.query(Attendance).filter(Attendance.date == today).all()

def get_absent(db):
    return db.query(Attendance).filter(Attendance.status == "absent").all()

# Persons CRUD
def add_person(db, name):
    person = Person(name=name)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person

def list_persons(db):
    return db.query(Person).all()

def get_person_by_name(db: Session, name: str):
    return db.query(Person).filter(Person.name == name).first()
