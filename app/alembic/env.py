from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine
import asyncio

from alembic import context

from app.models.performance import Performance

from sqlmodel import SQLModel

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        user_module_prefix='sqlmodel.sql.sqltypes.',
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """The actual migration execution, properly wrapped."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        user_module_prefix='sqlmodel.sql.sqltypes.',
        render_as_batch=True,
    )
    context.run_migrations()


async def run_async_migrations():
    """Run migrations in 'online' mode with async support."""
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        future=True
    )
    
    async with connectable.begin() as connection:
        await connection.run_sync(do_run_migrations)
        await connection.commit()

def run_migrations_online():
    """Run the async migrations."""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
