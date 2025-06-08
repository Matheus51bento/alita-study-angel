import pandas as pd
import os
import joblib
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.performance import Performance
from app.cron.ranker import treinar_modelos_para_todos_os_alunos
from app.schemas.recommendation import RecommendationInput, RecommendationOutput, IRTRecommendationOutput
from app.schemas.performance  import PerformanceCreate
from app.utils import irt_prioridade

from fastapi import APIRouter, HTTPException, Depends
from app.db.database import get_session

recomendacao_router = APIRouter()
@recomendacao_router.post("/recommendations/ranker/", response_model=list[RecommendationOutput])
async def recomendar_materias(
    body: RecommendationInput,
    session: AsyncSession = Depends(get_session)
):
    modelo_path = f"modelos/student_{body.student_id}/model.pkl"
    if not os.path.exists(modelo_path):
        raise HTTPException(status_code=404, detail="Modelo não encontrado para o aluno")

    model = joblib.load(modelo_path)

    query = await session.execute(
        select(Performance).where(Performance.student_id == body.student_id)
    )
    resultados = query.scalars().all()
    if not resultados:
        raise HTTPException(status_code=404, detail="Sem dados para o aluno")

    df = pd.DataFrame([r.__dict__ for r in resultados])
    df["score"] = 1 - ((df["desempenho"] + 1) / 2)
    df = df.sort_values("timestamp")  # ordena cronologicamente
    df = df.drop_duplicates(subset=["classe", "subclasse"], keep="last")

    if df.empty:
        raise HTTPException(status_code=404, detail="Sem performances recentes para recomendar")

    X_pred = df[["peso_classe", "peso_subclasse", "peso_por_questao"]]
    df["score_predito"] = model.predict(X_pred)

    recomendados = (
        df
        .sort_values("score_predito", ascending=False)
        [["classe", "subclasse", "desempenho", "score_predito"]]
        .reset_index(drop=True)
    )

    return [RecommendationOutput(**row) for row in recomendados.to_dict(orient="records")]

@recomendacao_router.get("/recommendations/ranker/treinar", tags=["ranker"])
async def treinar_modelos_endpoint(session: AsyncSession = Depends(get_session)):
    """
    Gatilho manual para treinar os modelos de todos os alunos.
    """
    await treinar_modelos_para_todos_os_alunos()
    return {"message": "Treinamento de modelos concluído com sucesso"}


@recomendacao_router.post("/recommendations/irt/", response_model=List[IRTRecommendationOutput])
async def recomendar_irt(payload: List[PerformanceCreate]):
    if not payload:
        raise HTTPException(status_code=400, detail="Payload vazio")

    df = pd.DataFrame([p.dict() for p in payload])
    df = irt_prioridade(df)

    agrupado = (
        df.groupby(["classe", "subclasse", "metrica"])
        .agg({"prioridade": "mean"})
        .reset_index()
        .sort_values("prioridade", ascending=False)
    )

    return [
        IRTRecommendationOutput(
            classe=row["classe"],
            subclasse=row["subclasse"],
            prioridade=row["prioridade"],
            metrica=row["metrica"]
        )
        for _, row in agrupado.iterrows()
    ]