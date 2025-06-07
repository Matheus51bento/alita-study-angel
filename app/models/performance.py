from sqlmodel import SQLModel, Field
from typing import Optional
import datetime

class Performance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: str = Field(index=True, nullable=False)
    classe: str = Field(nullable=False)
    subclasse: str = Field(nullable=False)
    desempenho: float = Field(nullable=False)
    peso_classe: float = Field(nullable=False)
    peso_subclasse: float = Field(nullable=False)
    peso_por_questao: float = Field(nullable=False)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, nullable=False)
