import os
import pickle
import pandas as pd
import joblib

from sqlalchemy.ext.asyncio import AsyncSession
from xgboost import XGBRanker
from sqlalchemy.future import select
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.models.performance import Performance
from app.db.engine import engine

BASE_DIR = "modelos"

async def carregar_dados_para_ranker(session: AsyncSession):
    query = await session.execute(select(Performance))
    resultados = query.scalars().all()

    df = pd.DataFrame([r.__dict__ for r in resultados])
    df = df.drop(columns=["_sa_instance_state", "id"])

    df["data"] = df["timestamp"].dt.date
    df["score"] = 1 - ((df["desempenho"] + 1) / 2)
    df = df.sort_values(["student_id", "data"])

    features = ["peso_classe", "peso_subclasse", "peso_por_questao"]
    X = df[features]
    y = df["score"]

    group = df.groupby(["student_id", "data"]).size().tolist()

    return X, y, group, df

def treinar_ranker(X, y, group):
    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        verbosity=1
    )
    model.fit(X, y, group=group)
    return model

def recomendar_para_aluno(model, df, student_id, data):
    filtro = (df["student_id"] == student_id) & (df["data"] == data)
    dados_aluno = df[filtro].copy()
    X_pred = dados_aluno[["peso_classe", "peso_subclasse", "peso_por_questao"]]

    # Gera predições
    dados_aluno["score_predito"] = model.predict(X_pred)

    # Agrupa por classe + subclasse e pega o melhor score
    dados_filtrados = (
        dados_aluno
        .sort_values("score_predito", ascending=False)
        .drop_duplicates(subset=["classe", "subclasse"])
    )

    return dados_filtrados.sort_values("score_predito", ascending=False)[
        ["classe", "subclasse", "score_predito"]
    ].reset_index(drop=True)

async def treinar_modelos_para_todos_os_alunos():
    async with AsyncSession(engine) as session:
        X, y, group, df = await carregar_dados_para_ranker(session)

        for (student_id, data), gsize in zip(df.groupby(["student_id", "data"]).groups.keys(), group):
            dados_aluno = df[(df["student_id"] == student_id) & (df["data"] == data)]
            if len(dados_aluno) < 3:
                continue  # não há dados suficientes

            X_aluno = dados_aluno[["peso_classe", "peso_subclasse", "peso_por_questao"]]
            y_aluno = dados_aluno["score"]

            model = treinar_ranker(X_aluno, y_aluno, group=[len(X_aluno)])

            path = f"modelos/student_{student_id}"
            os.makedirs(path, exist_ok=True)
            joblib.dump(model, f"{path}/model.pkl")
