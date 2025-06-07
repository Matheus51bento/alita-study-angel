from fastapi import APIRouter, HTTPException, Request, Depends
from app.schemas.performance import PerformanceCreate, PerformanceRead
from app.db.database import get_session
from app.models.performance import Performance
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from fastapi import Query
from typing import Optional

performance_router = APIRouter()


@performance_router.post("/performance/", response_model=PerformanceRead)
async def create_performance(
    performance: PerformanceCreate, session: AsyncSession = Depends(get_session)
):
    """
    Create a new performance record.
    """

    db_performance = PerformanceCreate(**performance.model_dump())
    session.add(db_performance)
    await session.commit()
    await session.refresh(db_performance)

    return PerformanceRead(**db_performance.model_dump())


@performance_router.post("/performance/batch/", response_model=list[PerformanceRead])
async def create_performance_batch(
    performances: list[PerformanceCreate], session: AsyncSession = Depends(get_session)
):
    """
    Create multiple performance records in a batch.
    """
    db_performances = [
        Performance(**performance.model_dump()) for performance in performances
    ]

    session.add_all(db_performances)
    await session.commit()

    for db_performance in db_performances:
        await session.refresh(db_performance)

    return [
        PerformanceRead(**db_performance.model_dump())
        for db_performance in db_performances
    ]


@performance_router.get(
    "/performance/{student_id}", response_model=list[PerformanceRead]
)
async def read_performances(
    student_id: str,
    classe: Optional[str] = Query(None),
    subclasse: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """
    Get a performance record by student ID.
    """
    stmt = select(Performance).where(Performance.student_id == student_id)

    if classe:
        stmt = stmt.where(Performance.classe == classe)
    if subclasse:
        stmt = stmt.where(Performance.subclasse == subclasse)

    result = await session.exec(stmt)
    performances = result.all()

    if not performances:
        raise HTTPException(status_code=404, detail="Performance not found")

    return performances
