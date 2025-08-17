from sqlalchemy.ext.asyncio import create_async_engine
from app.settings.config import settings

engine = create_async_engine("postgresql+asyncpg://user:password@localhost:5432/alita_db", echo=True)
