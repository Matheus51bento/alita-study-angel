
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.cron.ranker import treinar_modelos_para_todos_os_alunos  # import adiado
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(treinar_modelos_para_todos_os_alunos, 'interval', hours=24)
    scheduler.start()
    yield
    await engine.dispose()

async def get_session():
    async with AsyncSession(engine) as session:
        yield session
