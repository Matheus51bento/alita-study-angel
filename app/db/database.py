
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.settings.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()

async def get_session():
    async with AsyncSession(engine) as session:
        yield session
