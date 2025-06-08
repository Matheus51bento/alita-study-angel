import pandas as pd
import os
import joblib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.performance import Performance
from app.cron.ranker import treinar_modelos_para_todos_os_alunos
from app.schemas.recommendation import RecommendationInput, RecommendationOutput

from fastapi import APIRouter, HTTPException, Depends
from app.db.database import get_session

recomendacao_router = APIRouter()

@recomendacao_router.post("/recommendations/", response_model=list[RecommendationOutput])
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
    df["data"] = df["timestamp"].dt.date

    dados_dia = df[df["data"] == body.data]
    if dados_dia.empty:
        raise HTTPException(status_code=404, detail="Sem dados para essa data")

    X_pred = dados_dia[["peso_classe", "peso_subclasse", "peso_por_questao"]]
    dados_dia["score_predito"] = model.predict(X_pred)

    recomendados = (
        dados_dia
        .sort_values("score_predito", ascending=False)
        .drop_duplicates(subset=["classe", "subclasse"])
        .sort_values("score_predito", ascending=False)
    )

    return [
        RecommendationOutput(
            classe=row["classe"],
            subclasse=row["subclasse"],
            desempenho=row["desempenho"],
            score_predito=row["score_predito"],
        )
        for _, row in recomendados.iterrows()
    ]


@recomendacao_router.get("/recommendations/ranker/treinar", tags=["ranker"])
async def treinar_modelos_endpoint(session: AsyncSession = Depends(get_session)):
    """
    Gatilho manual para treinar os modelos de todos os alunos.
    """
    await treinar_modelos_para_todos_os_alunos()
    return {"message": "Treinamento de modelos concluído com sucesso"}
