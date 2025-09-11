"""
FastAPI application for Diabetes Health Indicators ML project.
Provides endpoints for data ingestion, model training, and predictions.
"""

import os
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import database and models
from api.db import SessionLocal
from api.models import DiabetesRaw, ModelRun
from api.schemas import DiabetesDataCreate, DiabetesDataResponse
from api.config import settings

# Import ML modules
try:
    from ml.infer import predict_one, explain_one, predict_and_explain, ValidationError, InferenceError
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML inference not available: {e}")
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# FastAPI app initialization
app = FastAPI(
    title="Diabetes Health Indicators API",
    description="API for diabetes risk prediction using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

class TrainRequest(BaseModel):
    retrain: bool = Field(default=False, description="Force retrain even if model exists")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set proportion")

class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    training_time: Optional[float] = None
    training_records: Optional[int] = None
    data_source: Optional[str] = None

class PredictRequest(BaseModel):
    # Using snake_case field names to match database model
    high_bp: int = Field(ge=0, le=1, description="High blood pressure (0=No, 1=Yes)")
    high_chol: int = Field(ge=0, le=1, description="High cholesterol (0=No, 1=Yes)")
    chol_check: int = Field(ge=0, le=1, description="Cholesterol check (0=No, 1=Yes)")
    bmi: float = Field(ge=10.0, le=100.0, description="Body Mass Index")
    smoker: int = Field(ge=0, le=1, description="Smoking status (0=No, 1=Yes)")
    stroke: int = Field(ge=0, le=1, description="Had stroke (0=No, 1=Yes)")
    heart_disease_or_attack: int = Field(ge=0, le=1, description="Heart disease or attack (0=No, 1=Yes)")
    phys_activity: int = Field(ge=0, le=1, description="Physical activity (0=No, 1=Yes)")
    fruits: int = Field(ge=0, le=1, description="Consume fruits (0=No, 1=Yes)")
    veggies: int = Field(ge=0, le=1, description="Consume vegetables (0=No, 1=Yes)")
    hvy_alcohol_consump: int = Field(ge=0, le=1, description="Heavy alcohol consumption (0=No, 1=Yes)")
    any_healthcare: int = Field(ge=0, le=1, description="Any healthcare coverage (0=No, 1=Yes)")
    no_docbc_cost: int = Field(ge=0, le=1, description="Could not see doctor because of cost (0=No, 1=Yes)")
    gen_hlth: int = Field(ge=1, le=5, description="General health (1=Excellent, 5=Poor)")
    ment_hlth: int = Field(ge=0, le=30, description="Mental health days in past 30 days")
    phys_hlth: int = Field(ge=0, le=30, description="Physical health days in past 30 days")
    diff_walking: int = Field(ge=0, le=1, description="Difficulty walking (0=No, 1=Yes)")
    sex: int = Field(ge=0, le=1, description="Sex (0=Female, 1=Male)")
    age: int = Field(ge=1, le=13, description="Age category (1-13)")
    education: int = Field(ge=1, le=6, description="Education level (1-6)")
    income: int = Field(ge=1, le=8, description="Income level (1-8)")

class PredictResponse(BaseModel):
    probability: float
    class_prediction: int
    risk_level: str
    confidence: float
    top_features: Optional[List[Dict[str, Any]]] = None
    explanation_available: bool
    patient_id: Optional[int] = None

class ResetRequest(BaseModel):
    confirm: bool = Field(default=False, description="Confirmation flag to prevent accidental resets")
    reset_data: bool = Field(default=True, description="Reset training data in database")
    reset_model: bool = Field(default=True, description="Reset trained model artifacts")

class ResetResponse(BaseModel):
    status: str
    message: str
    deleted_records: int = 0
    deleted_models: int = 0
    artifacts_removed: List[str] = []

class MetricsResponse(BaseModel):
    metrics: Optional[Dict[str, Any]]
    model_info: Optional[Dict[str, Any]]
    last_updated: Optional[datetime]

# Dependency to get database session
def get_database():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Validation failed", "details": str(exc)}
    )

@app.exception_handler(InferenceError)
async def inference_exception_handler(request, exc):
    logger.error(f"Inference error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Prediction failed", "details": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": "An unexpected error occurred"}
    )


# Utility functions
def load_metrics_json() -> Optional[Dict[str, Any]]:
    """Load metrics from the saved JSON file."""
    metrics_path = project_root / "ml" / "artifacts" / "metrics.json"
    
    if not metrics_path.exists():
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        return None

def get_model_info() -> Dict[str, Any]:
    """Get information about the current model."""
    model_path = project_root / "ml" / "artifacts" / "model.pkl"
    metrics_path = project_root / "ml" / "artifacts" / "metrics.json"
    
    info = {
        "model_exists": model_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path)
    }
    
    if model_path.exists():
        stat = model_path.stat()
        info["model_size"] = stat.st_size
        info["model_modified"] = datetime.fromtimestamp(stat.st_mtime)
    
    if metrics_path.exists():
        stat = metrics_path.stat()
        info["metrics_modified"] = datetime.fromtimestamp(stat.st_mtime)
        
    return info


# ================================
# API ENDPOINTS
# ================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        HealthResponse: Service status and metadata
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(),
        version=app.version
    )


@app.post("/ingest/record", response_model=DiabetesDataResponse, tags=["Data Ingestion"])
async def ingest_diabetes_record(
    record: DiabetesDataCreate,
    db = Depends(get_database)
):
    """
    Ingest a new diabetes health indicators record into the database.
    
    Args:
        record: Diabetes health indicators record data
        db: Database session
        
    Returns:
        DiabetesDataResponse: Created record with ID
        
    Raises:
        HTTPException: If validation or persistence fails
    """
    logger.info(f"Ingesting new diabetes record")
    
    try:
        # Create new record
        db_record = DiabetesRaw(**record.dict())
        
        # Add to database
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        logger.info(f"Successfully ingested record with ID: {db_record.id}")
        
        return DiabetesDataResponse(
            id=db_record.id,
            **record.dict(),
            created_at=db_record.created_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to ingest record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest record: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse, tags=["Model Training"])  
async def train_model(
    request: TrainRequest = TrainRequest(),
    db = Depends(get_database)
):
    """
    Train the diabetes prediction model.
    
    Args:
        request: Training configuration
        db: Database session
        
    Returns:
        TrainResponse: Training results and metrics
        
    Raises:
        HTTPException: If training fails
    """
    logger.info(f"Training request received: retrain={request.retrain}, test_size={request.test_size}")
    
    # Check if model already exists
    model_path = project_root / "ml" / "artifacts" / "model.pkl"
    if model_path.exists() and not request.retrain:
        logger.info("Model already exists and retrain=False")
        
        # Load existing metrics
        metrics = load_metrics_json()
        
        return TrainResponse(
            status="success",
            message="Model already trained. Use retrain=true to force retraining.",
            metrics=metrics,
            model_id=f"existing_model_{int(model_path.stat().st_mtime)}"
        )
    
    try:
        # Record training start
        start_time = datetime.now()
        
        # Build training command
        train_script = project_root / "ml" / "train.py"
        cmd = [
            sys.executable, 
            str(train_script),
            "--test_size", str(request.test_size)
        ]
        
        logger.info(f"Starting training with command: {' '.join(cmd)}")
        
        # Run training script
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Training failed: {result.stderr}"
            )
        
        logger.info(f"Training completed successfully in {training_time:.2f}s")
        
        # Load metrics from the training output
        metrics = load_metrics_json()
        
        if not metrics:
            logger.warning("Metrics file not found after training")
        
        # Save training run to database
        model_run = ModelRun(
            auc=metrics.get("auc", 0.0) if metrics else 0.0,
            f1=metrics.get("f1", 0.0) if metrics else 0.0,
            recall=metrics.get("recall", 0.0) if metrics else 0.0,
            precision=metrics.get("precision", 0.0) if metrics else 0.0,
            notes=f"XGBoost model training completed in {training_time:.2f}s"
        )
        
        db.add(model_run)
        db.commit()
        db.refresh(model_run)
        
        # Extract data info from metrics if available
        training_records = None
        data_source = None
        if metrics and 'data_info' in metrics:
            data_info = metrics['data_info']
            training_records = data_info.get('train_records')
            data_source = data_info.get('source')
        
        return TrainResponse(
            status="success",
            message="Model trained successfully",
            metrics=metrics,
            model_id=str(model_run.id),
            training_time=training_time,
            training_records=training_records,
            data_source=data_source
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Training timeout exceeded")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Training timeout exceeded (5 minutes)"
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/reset", response_model=ResetResponse, tags=["System Management"])
async def reset_system(
    request: ResetRequest = ResetRequest(),
    db = Depends(get_database)
):
    """
    Reset system by clearing data and/or model artifacts.
    
    Args:
        request: Reset configuration with confirmation
        db: Database session
        
    Returns:
        ResetResponse: Reset operation results
        
    Raises:
        HTTPException: If reset fails or not confirmed
    """
    logger.info(f"Reset request received: confirm={request.confirm}, reset_data={request.reset_data}, reset_model={request.reset_model}")
    
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset operation requires explicit confirmation. Set 'confirm=true'"
        )
    
    deleted_records = 0
    deleted_models = 0
    artifacts_removed = []
    
    try:
        # Reset database data
        if request.reset_data:
            # Clear training data
            deleted_records = db.query(DiabetesRaw).count()
            db.query(DiabetesRaw).delete()
            
            # Clear model runs
            deleted_models = db.query(ModelRun).count()
            db.query(ModelRun).delete()
            
            db.commit()
            logger.info(f"Cleared {deleted_records} data records and {deleted_models} model runs")
        
        # Reset model artifacts
        if request.reset_model:
            artifacts_dir = project_root / "ml" / "artifacts"
            
            # Remove model file
            model_path = artifacts_dir / "model.pkl"
            if model_path.exists():
                model_path.unlink()
                artifacts_removed.append("model.pkl")
                logger.info("Removed model.pkl")
            
            # Remove metrics file
            metrics_path = artifacts_dir / "metrics.json"
            if metrics_path.exists():
                metrics_path.unlink()
                artifacts_removed.append("metrics.json")
                logger.info("Removed metrics.json")
            
            # Remove any other artifacts
            for artifact_file in artifacts_dir.glob("*.pkl"):
                if artifact_file.name != "model.pkl":  # Already handled above
                    artifact_file.unlink()
                    artifacts_removed.append(artifact_file.name)
                    logger.info(f"Removed {artifact_file.name}")
            
            for artifact_file in artifacts_dir.glob("*.json"):
                if artifact_file.name != "metrics.json":  # Already handled above
                    artifact_file.unlink()
                    artifacts_removed.append(artifact_file.name)
                    logger.info(f"Removed {artifact_file.name}")
        
        return ResetResponse(
            status="success",
            message="System reset completed successfully",
            deleted_records=deleted_records,
            deleted_models=deleted_models,
            artifacts_removed=artifacts_removed
        )
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_diabetes_risk(
    patient: PredictRequest,
    include_explanation: bool = True,
    top_k: int = 3,
    threshold: float = 0.5,
    save_to_db: bool = False,
    db = Depends(get_database)
):
    """
    Predict diabetes risk for a patient.
    
    Args:
        patient: Patient data for prediction
        include_explanation: Whether to include SHAP explanations
        top_k: Number of top features to explain
        threshold: Classification threshold
        save_to_db: Whether to save prediction to database
        db: Database session
        
    Returns:
        PredictResponse: Prediction results with optional explanations
        
    Raises:
        HTTPException: If prediction fails or ML not available
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML inference not available. Please install required dependencies."
        )
    
    logger.info(f"Prediction request for patient: bmi={patient.bmi}, age={patient.age}, sex={patient.sex}")
    
    # Convert patient data to dict and map field names to Kaggle dataset format
    patient_data = patient.dict()
    
    # Map snake_case API fields to Kaggle dataset field names
    field_mapping = {
        'high_bp': 'HighBP',
        'high_chol': 'HighChol', 
        'chol_check': 'CholCheck',
        'bmi': 'BMI',
        'smoker': 'Smoker',
        'stroke': 'Stroke',
        'heart_disease_or_attack': 'HeartDiseaseorAttack',
        'phys_activity': 'PhysActivity',
        'fruits': 'Fruits',
        'veggies': 'Veggies',
        'hvy_alcohol_consump': 'HvyAlcoholConsump',
        'any_healthcare': 'AnyHealthcare',
        'no_docbc_cost': 'NoDocbcCost',
        'gen_hlth': 'GenHlth',
        'ment_hlth': 'MentHlth',
        'phys_hlth': 'PhysHlth',
        'diff_walking': 'DiffWalk',
        'sex': 'Sex',
        'age': 'Age',
        'education': 'Education',
        'income': 'Income'
    }
    
    # Convert field names for ML inference
    mapped_data = {field_mapping[k]: v for k, v in patient_data.items()}
    
    try:
        if include_explanation:
            # Get prediction with explanation
            result = predict_and_explain(
                payload=mapped_data,
                threshold=threshold,
                top_k=top_k
            )
            
            top_features = result.get('explanations', [])
            explanation_available = result.get('explanation_available', False)
            
        else:
            # Get prediction only
            result = predict_one(payload=mapped_data, threshold=threshold)
            top_features = None
            explanation_available = False
        
        # Optional: Save to database
        patient_id = None
        if save_to_db:
            try:
                # Create record in diabetes_raw table
                db_record = DiabetesRaw(
                    **patient_data,
                    diabetes_binary=result['class']  # Save predicted class as target
                )
                db.add(db_record)
                db.commit()
                db.refresh(db_record)
                patient_id = db_record.id
                
                logger.info(f"Saved prediction to database with ID: {patient_id}")
            except Exception as e:
                logger.warning(f"Failed to save prediction to database: {e}")
                # Don't fail the prediction if DB save fails
        
        logger.info(f"Prediction successful: class={result['class']}, probability={result['probability']:.3f}")
        
        return PredictResponse(
            probability=result['probability'],
            class_prediction=result['class'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            top_features=top_features,
            explanation_available=explanation_available,
            patient_id=patient_id
        )
        
    except ValidationError as e:
        logger.error(f"Validation error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid patient data: {str(e)}"
        )
    except InferenceError as e:
        logger.error(f"Inference error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model Info"])
async def get_model_metrics():
    """
    Get current model metrics and information.
    
    Returns:
        MetricsResponse: Model metrics and metadata
    """
    logger.info("Metrics request received")
    
    # Load metrics from file
    metrics = load_metrics_json()
    
    # Get model information
    model_info = get_model_info()
    
    # Get last updated timestamp
    last_updated = None
    if model_info.get("metrics_modified"):
        last_updated = model_info["metrics_modified"]
    
    logger.info(f"Returning metrics: available={metrics is not None}")
    
    return MetricsResponse(
        metrics=metrics,
        model_info=model_info,
        last_updated=last_updated
    )


# Additional utility endpoints
@app.get("/", tags=["Info"])
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Diabetes Health Indicators Prediction API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "ingest": "POST /ingest/record",
            "train": "POST /train", 
            "predict": "POST /predict",
            "metrics": "GET /metrics"
        }
    }


@app.get("/status", tags=["Info"])
async def get_system_status():
    """Get system status including model availability."""
    model_info = get_model_info()
    
    return {
        "api_status": "running",
        "ml_available": ML_AVAILABLE,
        "model_trained": model_info["model_exists"],
        "metrics_available": model_info["metrics_exists"],
        "database_connected": True,  # If we got here, DB is working
        "timestamp": datetime.now()
    }


@app.get("/records", tags=["Data"])
async def get_recent_records(
    limit: int = 50,
    offset: int = 0,
    db = Depends(get_database)
):
    """
    Get recent diabetes records from the database.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        db: Database session
        
    Returns:
        List of recent records with metadata
    """
    try:
        # Query recent records
        records = db.query(DiabetesRaw)\
                   .order_by(DiabetesRaw.created_at.desc())\
                   .offset(offset)\
                   .limit(limit)\
                   .all()
        
        # Get total count
        total_count = db.query(DiabetesRaw).count()
        
        # Convert to response format
        records_data = []
        for record in records:
            record_dict = {
                "id": record.id,
                "high_bp": record.high_bp,
                "high_chol": record.high_chol,
                "chol_check": record.chol_check,
                "bmi": record.bmi,
                "smoker": record.smoker,
                "stroke": record.stroke,
                "heart_disease_or_attack": record.heart_disease_or_attack,
                "phys_activity": record.phys_activity,
                "fruits": record.fruits,
                "veggies": record.veggies,
                "hvy_alcohol_consump": record.hvy_alcohol_consump,
                "any_healthcare": record.any_healthcare,
                "no_docbc_cost": record.no_docbc_cost,
                "gen_hlth": record.gen_hlth,
                "ment_hlth": record.ment_hlth,
                "phys_hlth": record.phys_hlth,
                "diff_walking": record.diff_walking,
                "sex": record.sex,
                "age": record.age,
                "education": record.education,
                "income": record.income,
                "diabetes_binary": record.diabetes_binary,
                "created_at": record.created_at
            }
            records_data.append(record_dict)
        
        logger.info(f"Retrieved {len(records_data)} records (offset={offset}, limit={limit})")
        
        return {
            "records": records_data,
            "total_count": total_count,
            "returned_count": len(records_data),
            "offset": offset,
            "limit": limit,
            "has_more": (offset + len(records_data)) < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve records: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve records: {str(e)}"
        )


# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Diabetes Health Indicators API")
    logger.info(f"ML inference available: {ML_AVAILABLE}")
    
    # Verify critical paths exist
    os.makedirs("ml/artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    model_info = get_model_info()
    logger.info(f"Model exists: {model_info['model_exists']}")
    logger.info(f"Metrics exists: {model_info['metrics_exists']}")


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Diabetes Health Indicators API")


# Main function for running the API
def main():
    """Run the FastAPI application with uvicorn."""
    
    # Configure uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()