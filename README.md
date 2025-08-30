# alita
Study angel

docker compose up -d db
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5433/alita_db uvicorn app.main:app --reload

## Gerar

python gerar_alunos.py <total_alunos>

python gerar_simulacao.py alunos_gerados/aluno_1000.csv 1000 --dias 50

## testes

pytest app/test.py

Coverage

sudo systemctl stop postgresql

## Docker

```bash
docker exec -it alita2-db-1 psql -U user -d alita_db
```

```bash
docker compose exec app alembic -c app/alembic.ini revision --autogenerate -m "initial"
```

```bash
docker compose exec app alembic -c app/alembic.ini upgrade head
```
