from pydantic import BaseModel
from typing import List


class PrevisaoRequest(BaseModel):
    aluno_id: int
    dados_aluno: List[List[float]]

class TreinamentoRequest(BaseModel):
    aluno_id: str