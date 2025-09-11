"""
Database connection and session management for Diabetes Health Indicators ML API.
Uses SQLAlchemy with PostgreSQL backend.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.API_DEBUG,
    pool_pre_ping=True,
    pool_recycle=300
)

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create Base class for ORM models
Base = declarative_base()


def get_db() -> Session:
    """
    Database session dependency for FastAPI.
    Yields a database session and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    # Import all models to ensure they are registered with SQLAlchemy
    from . import models  # noqa: F401
    
    # Create all tables
    Base.metadata.create_all(bind=engine)