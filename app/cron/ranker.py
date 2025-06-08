import os
import joblib
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

        total = len(dados_por_aluno)
        sucesso, falha = 0, 0

        for student_id, dados in dados_por_aluno.items():
            try:
                logger.info(f"Iniciando treino para aluno {student_id}")
                model = treinar_ranker(dados["X"], dados["y"], group=dados["group"])

                path = f"{BASE_DIR}/student_{student_id}"
                os.makedirs(path, exist_ok=True)
                joblib.dump(model, f"{path}/model.pkl")

                logger.info(f"Modelo salvo com sucesso para aluno {student_id}")
                sucesso += 1

            except Exception as e:
                logger.error(f"Erro ao treinar modelo para aluno {student_id}: {e}")
                falha += 1

        logger.info("Treinamento concluído.")
        logger.info(f"Total de alunos processados: {total}")
        logger.info(f"Modelos treinados com sucesso: {sucesso}")
        logger.info(f"Falhas no treinamento: {falha}")
