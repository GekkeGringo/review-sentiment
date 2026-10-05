import os
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    case,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(16))
    p_bad: Mapped[float] = mapped_column(Float)
    p_neutral: Mapped[float] = mapped_column(Float)
    p_good: Mapped[float] = mapped_column(Float)


class Store:
    def __init__(self, url=None):
        url = url or os.getenv("DATABASE_URL", "sqlite:///reviews.db")
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def save(self, text, label, probs):
        with self.Session() as s:
            s.add(
                Review(
                    text=text,
                    label=label,
                    p_bad=probs["Bad"],
                    p_neutral=probs["Neutral"],
                    p_good=probs["Good"],
                )
            )
            s.commit()

    def daily_negative_share(self):
        day = func.date(Review.created_at)
        bad = func.sum(case((Review.label == "Bad", 1), else_=0))
        query = select(day, func.count(), bad).group_by(day).order_by(day)
        with self.Session() as s:
            rows = s.execute(query).all()
        return [
            {"date": str(d), "total": t, "bad_share": round(float(b) / t, 3)}
            for d, t, b in rows
        ]

    def most_negative(self, days=7, limit=10):
        since = utcnow() - timedelta(days=days)
        query = (
            select(Review)
            .where(Review.created_at >= since)
            .order_by(Review.p_bad.desc())
            .limit(limit)
        )
        with self.Session() as s:
            rows = s.scalars(query).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "p_bad": r.p_bad,
                "text": r.text[:300],
            }
            for r in rows
        ]