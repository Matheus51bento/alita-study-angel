from fastapi import FastAPI, HTTPException, Request

# from app.utils import calculate_elbow_method, calculate_priorities
from app.db.database import get_session, lifespan
# from app.schemas.content import Conteudo

from app.routers.performance import performance_router

# from sklearn.cluster import KMeans
# import pandas as pd
import logging

from typing import List

app = FastAPI(lifespan=lifespan)

app.include_router(performance_router, prefix="/api/v1", tags=["performance"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @app.middleware("http")
# async def log_request_payload(request: Request, call_next):
#     body = await request.body()
#     logger.info("Payload recebido: %s", body.decode("utf-8"))
#     response = await call_next(request)
#     return response


# @app.post("/contents/")
# def return_contents(conteudos: List[Conteudo], number_or_contents: int = 7):

#     if not conteudos:
#         raise HTTPException(status_code=400, detail="Nenhum conteúdo foi enviado.")

#     df= pd.DataFrame([conteudo.model_dump() for conteudo in conteudos])

#     X = df[['Desempenho', 'Peso_da_classe', 'Peso_da_subclasse', 'Peso_por_questao']]

#     n_clusters = calculate_elbow_method(df)

#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X)

#     labels = kmeans.labels_

#     df['Cluster'] = labels

#     return calculate_priorities(df, number_or_contents).to_dict(orient='records')
