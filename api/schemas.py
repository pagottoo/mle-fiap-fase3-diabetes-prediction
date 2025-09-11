"""
Pydantic schemas for Diabetes Risk ML API.
Defines request/response models with validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class DiabetesDataBase(BaseModel):
    """Base schema for diabetes health indicators data."""
    
    # Health indicators
    high_bp: int = Field(..., ge=0, le=1, description="High blood pressure (0: no, 1: yes)")
    high_chol: int = Field(..., ge=0, le=1, description="High cholesterol (0: no, 1: yes)")
    chol_check: int = Field(..., ge=0, le=1, description="Cholesterol check in past 5 years (0: no, 1: yes)")
    bmi: float = Field(..., ge=12.0, le=50.0, description="Body Mass Index")
    smoker: int = Field(..., ge=0, le=1, description="Smoked at least 100 cigarettes (0: no, 1: yes)")
    stroke: int = Field(..., ge=0, le=1, description="Ever had stroke (0: no, 1: yes)")
    heart_disease_or_attack: int = Field(..., ge=0, le=1, description="Coronary heart disease or MI (0: no, 1: yes)")
    phys_activity: int = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0: no, 1: yes)")
    fruits: int = Field(..., ge=0, le=1, description="Consume fruit 1+ times per day (0: no, 1: yes)")
    veggies: int = Field(..., ge=0, le=1, description="Consume vegetables 1+ times per day (0: no, 1: yes)")
    hvy_alcohol_consump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption (0: no, 1: yes)")
    any_healthcare: int = Field(..., ge=0, le=1, description="Any healthcare coverage (0: no, 1: yes)")
    no_docbc_cost: int = Field(..., ge=0, le=1, description="Couldn't see doctor due to cost (0: no, 1: yes)")
    gen_hlth: int = Field(..., ge=1, le=5, description="General health (1: excellent, 5: poor)")
    ment_hlth: int = Field(..., ge=0, le=30, description="Days mental health not good (0-30)")
    phys_hlth: int = Field(..., ge=0, le=30, description="Days physical health not good (0-30)")
    diff_walking: int = Field(..., ge=0, le=1, description="Difficulty walking/climbing stairs (0: no, 1: yes)")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    age: int = Field(..., ge=1, le=13, description="Age category (1: 18-24, 13: 80+)")
    education: int = Field(..., ge=1, le=6, description="Education level (1-6)")
    income: int = Field(..., ge=1, le=8, description="Income level (1-8)")


class DiabetesDataCreate(DiabetesDataBase):
    """Schema for creating diabetes data records."""
    
    diabetes_binary: Optional[int] = Field(None, ge=0, le=1, description="Diabetes diagnosis (0: no, 1: yes)")


class DiabetesDataPredict(DiabetesDataBase):
    """Schema for diabetes prediction requests."""
    
    pass


class DiabetesDataResponse(DiabetesDataBase):
    """Schema for diabetes data responses."""
    
    id: int
    diabetes_binary: Optional[int] = None
    
    class Config:
        from_attributes = True  # For Pydantic v2 compatibility


class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    
    data: DiabetesDataPredict
    include_explanation: bool = Field(True, description="Include SHAP explanation in response")


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    
    prediction: int = Field(..., description="Predicted class (0: no diabetes, 1: diabetes)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of diabetes")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    explanation: Optional[dict] = Field(None, description="SHAP explanation if requested")


class ModelRunCreate(BaseModel):
    """Schema for creating model run records."""
    
    auc: Optional[float] = Field(None, ge=0.0, le=1.0, description="ROC AUC score")
    f1: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall score")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision score")
    notes: Optional[str] = Field(None, description="Additional notes about the model run")


class ModelRunResponse(BaseModel):
    """Schema for model run responses."""
    
    id: int
    created_at: str
    auc: Optional[float] = None
    f1: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    notes: Optional[str] = None
    
    class Config:
        from_attributes = True  # For Pydantic v2 compatibility


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(..., description="Service status")
    database_connected: bool = Field(..., description="Database connection status")
    model_loaded: bool = Field(..., description="ML model loading status")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")