from fastapi import FastAPI
from app.db.database import lifespan
from app.routers.performance import performance_router
from app.routers.ranker import recomendacao_router

import logging
from typing import List

# Configuração de logging global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Log detalhado apenas para o APScheduler
logging.getLogger("apscheduler").setLevel(logging.DEBUG)

# Logger da aplicação principal (opcional)
logger = logging.getLogger(__name__)

# Instância FastAPI com ciclo de vida (lifespan)
app = FastAPI(lifespan=lifespan)

# Rotas da aplicação
app.include_router(performance_router, prefix="/api/v1", tags=["performance"])
app.include_router(recomendacao_router, prefix="/api/v1", tags=["ranker"])

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.middleware("http")
# async def log_request_payload(request: Request, call_next):
#     body = await request.body()
#     logger.info("Payload recebido: %s", body.decode("utf-8"))
#     response = await call_next(request)
#     return response
