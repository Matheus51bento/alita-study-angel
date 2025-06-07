from sqlmodel import SQLModel
from datetime import datetime

class PerformanceCreate(SQLModel):
    student_id: str
    classe: str
    subclasse: str
    desempenho: float
    peso_classe: float
    peso_subclasse: float
    peso_por_questao: float

class PerformanceRead(PerformanceCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True