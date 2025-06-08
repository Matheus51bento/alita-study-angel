from pydantic import BaseModel
from datetime import date

class RecommendationInput(BaseModel):
    student_id: str

class RecommendationOutput(BaseModel):
    classe: str
    subclasse: str
    desempenho: float
    score_predito: float


class IRTRecommendationOutput(BaseModel):
    classe: str
    subclasse: str
    prioridade: float
    metrica: str