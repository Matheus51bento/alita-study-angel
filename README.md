# alita
Study angel

uvicorn main:app --reload

testes

pytest app/test.py

Coverage


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
