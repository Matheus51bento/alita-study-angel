from pydantic import BaseModel

class Conteudo(BaseModel):
    ID: int
    Classe: str
    Subclasse: str
    Desempenho: float
    Peso_da_classe: float
    Peso_da_subclasse: float
    Peso_por_questao: float