from pydantic import BaseModel

class AttendanceCreate(BaseModel):
    person_name: str
    date: str
    time: str
    status: str            
    confidence: float
    source: str            

class AttendanceRead(AttendanceCreate):
    id: int

    class Config:
        from_attributes = True

class PersonAdd(BaseModel):
    name: str
    output_dir: str    

class PersonRead(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True
