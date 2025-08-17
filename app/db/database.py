
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import OperationalError, DBAPIError
from sqlalchemy import text
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.engine import engine

import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Função de verificação de conexão com o banco de dados
async def verificar_conexao():
    try:
        async with AsyncSession(engine) as session:
            # Use text() para garantir que o SQL seja tratado corretamente
            result = await session.exec(text("SELECT 1"))
            if result:
                logger.info("Conexão com o banco de dados bem-sucedida.")
            else:
                logger.error("Falha ao conectar com o banco de dados.")
                raise Exception("Não foi possível conectar ao banco de dados.")
    except DBAPIError as e:
        logger.error(f"Erro na conexão com o banco de dados: {e}")
        raise Exception(f"Erro ao conectar com o banco de dados: {e}")
    except Exception as e:
        logger.error(f"Erro ao conectar com o banco de dados: {e}")
        raise Exception(f"Erro ao conectar com o banco de dados: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.cron.ranker import treinar_modelos_para_todos_os_alunos  # import adiado
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    logger.info("Verificando conexão com o banco de dados antes de iniciar...")
    await verificar_conexao()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(treinar_modelos_para_todos_os_alunos, 'interval', hours=24)
    scheduler.start()
    yield
    await engine.dispose()

async def get_session():
    async with AsyncSession(engine) as session:
        yield session
