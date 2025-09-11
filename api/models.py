"""
SQLAlchemy ORM models for Diabetes Risk ML API.
Defines database table structures and relationships.
"""

from sqlalchemy import Column, Integer, Float, Text, DateTime
from sqlalchemy.sql import func
from .db import Base


class DiabetesRaw(Base):
    """Diabetes health indicators raw data model."""
    
    __tablename__ = "diabetes_raw"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Health indicators
    high_bp = Column(Integer, nullable=False)  # 0: no, 1: yes - high blood pressure
    high_chol = Column(Integer, nullable=False)  # 0: no, 1: yes - high cholesterol
    chol_check = Column(Integer, nullable=False)  # 0: no, 1: yes - cholesterol check in 5 years
    bmi = Column(Float, nullable=False)  # Body Mass Index
    smoker = Column(Integer, nullable=False)  # 0: no, 1: yes - smoked at least 100 cigarettes
    stroke = Column(Integer, nullable=False)  # 0: no, 1: yes - ever had stroke
    heart_disease_or_attack = Column(Integer, nullable=False)  # 0: no, 1: yes - coronary heart disease/MI
    phys_activity = Column(Integer, nullable=False)  # 0: no, 1: yes - physical activity in past 30 days
    fruits = Column(Integer, nullable=False)  # 0: no, 1: yes - consume fruit 1+ times per day
    veggies = Column(Integer, nullable=False)  # 0: no, 1: yes - consume vegetables 1+ times per day
    hvy_alcohol_consump = Column(Integer, nullable=False)  # 0: no, 1: yes - heavy alcohol consumption
    any_healthcare = Column(Integer, nullable=False)  # 0: no, 1: yes - any healthcare coverage
    no_docbc_cost = Column(Integer, nullable=False)  # 0: no, 1: yes - couldn't see doctor due to cost
    gen_hlth = Column(Integer, nullable=False)  # 1-5 scale general health (1=excellent, 5=poor)
    ment_hlth = Column(Integer, nullable=False)  # 0-30 days mental health not good
    phys_hlth = Column(Integer, nullable=False)  # 0-30 days physical health not good
    diff_walking = Column(Integer, nullable=False)  # 0: no, 1: yes - difficulty walking/climbing stairs
    sex = Column(Integer, nullable=False)  # 0: female, 1: male
    age = Column(Integer, nullable=False)  # 1-13 age categories (1: 18-24, 13: 80+)
    education = Column(Integer, nullable=False)  # 1-6 education level
    income = Column(Integer, nullable=False)  # 1-8 income scale
    
    # Target variable
    diabetes_binary = Column(Integer)  # 0: no diabetes, 1: diabetes


class ModelRun(Base):
    """Model training run metadata."""
    
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    auc = Column(Float)  # ROC AUC score
    f1 = Column(Float)   # F1 score
    recall = Column(Float)  # Recall score
    precision = Column(Float)  # Precision score
    notes = Column(Text)  # Additional notes about the model run