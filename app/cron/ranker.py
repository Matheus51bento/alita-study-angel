import os
import joblib
import numpy as np
import logging
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.engine import engine
from app.models.performance import Performance
from xgboost import XGBRanker
from sqlalchemy.future import select

# Setup de logging
logger = logging.getLogger("ranker_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = "modelos"

def dcg(relevancias, k):
    relevancias = np.asarray(relevancias)[:k]
    if relevancias.size:
        return relevancias[0] + np.sum(relevancias[1:] / np.log2(np.arange(2, relevancias.size + 1)))
    return 0.0

def ndcg(y_true, y_pred, k=5):
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order)
    dcg_max = dcg(sorted(y_true, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg(y_true_sorted, k) / dcg_max

async def carregar_dados_para_ranker(session: AsyncSession):
    query = await session.execute(select(Performance))
    resultados = query.scalars().all()

    if not resultados:
        logger.warning("Nenhum dado de performance encontrado.")
        return {}

    df = pd.DataFrame([r.__dict__ for r in resultados])
    df = df.drop(columns=["_sa_instance_state", "id"])

    df["data"] = df["timestamp"].dt.date
    df["score"] = 1 - ((df["desempenho"] + 1) / 2)
    df = df.sort_values(["student_id", "data"])

    features = ["peso_classe", "peso_subclasse", "peso_por_questao"]

    dados_por_aluno = {}
    for student_id, df_aluno in df.groupby("student_id"):
        if len(df_aluno) < 5:
            logger.info(f"Aluno {student_id} ignorado (menos de 5 registros)")
            continue

        X = df_aluno[features]
        y = df_aluno["score"]
        group = df_aluno.groupby("data").size().tolist()

        dados_por_aluno[student_id] = {
            "X": X,
            "y": y,
            "group": group,
            "df": df_aluno.reset_index(drop=True)
        }

    return dados_por_aluno


def treinar_ranker(X, y, group):
    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        verbosity=0  # silêncio interno do XGBoost
    )
    model.fit(X, y, group=group)
    return model

async def treinar_modelos_para_todos_os_alunos():
    async with AsyncSession(engine) as session:
        dados_por_aluno = await carregar_dados_para_ranker(session)

        for student_id, dados in dados_por_aluno.items():
            X = dados["X"]
            y = dados["y"]
            group = dados["group"]
            df_aluno = dados["df"]

            if len(X) < 5:
                continue

            model = treinar_ranker(X, y, group)

            # Avaliação usando NDCG
            y_pred = model.predict(X)
            score_ndcg = ndcg(y_true=y.values, y_pred=y_pred, k=5)
            logger.info(f"[Aluno {student_id}] NDCG@5 = {score_ndcg:.4f}")

            # Salvar modelo
            path = f"modelos/student_{student_id}"
            os.makedirs(path, exist_ok=True)
            joblib.dump(model, f"{path}/model.pkl")