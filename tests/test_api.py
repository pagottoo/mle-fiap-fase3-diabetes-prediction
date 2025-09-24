"""
Unit tests for the FastAPI application.
"""

import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the FastAPI app
from api.main import app
from api.models import DiabetesHealthRecord, ModelRun

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns correct response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"


class TestRootEndpoints:
    """Test root and info endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert len(data["endpoints"]) >= 5
    
    def test_status_endpoint(self):
        """Test status endpoint returns system status."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "api_status" in data
        assert "ml_available" in data
        assert "model_trained" in data
        assert "metrics_available" in data
        assert "database_connected" in data
        assert "timestamp" in data


class TestDataIngestion:
    """Test data ingestion endpoint."""
    
    @patch('api.main.get_database')
    def test_ingest_valid_record(self, mock_get_db):
        """Test ingesting a valid diabetes health record."""
        # Mock database session
        mock_db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_db
        mock_get_db.return_value.__exit__.return_value = None
        
        # Mock database record
        mock_record = MagicMock()
        mock_record.id = 123
        mock_record.created_at = "2025-10-01T12:00:00"
        mock_db.refresh.return_value = mock_record
        
        # Valid patient data
        valid_data = {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1, "target": 1
        }
        
        response = client.post("/ingest/record", json=valid_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return the record with ID
        assert "id" in data or response.status_code in [500, 503]  # May fail without real DB
    
    def test_ingest_invalid_record(self):
        """Test ingesting invalid data fails validation."""
        invalid_data = {
            "age": 150,  # Invalid age
            "sex": 1,
            "cp": 3
            # Missing required fields
        }
        
        response = client.post("/ingest/record", json=invalid_data)
        
        # Should fail validation
        assert response.status_code == 422  # Validation error


class TestModelTraining:
    """Test model training endpoint."""
    
    @patch('api.main.subprocess.run')
    @patch('api.main.load_metrics_json')
    @patch('api.main.get_database')
    def test_train_model_success(self, mock_get_db, mock_load_metrics, mock_subprocess):
        """Test successful model training."""
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Mock metrics
        mock_metrics = {
            "accuracy": 0.85,
            "roc_auc": 0.92,
            "f1_score": 0.83
        }
        mock_load_metrics.return_value = mock_metrics
        
        # Mock database
        mock_db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_db
        mock_get_db.return_value.__exit__.return_value = None
        
        # Mock model run record
        mock_model_run = MagicMock()
        mock_model_run.version = "v_1728000000"
        mock_db.refresh.return_value = mock_model_run
        
        response = client.post("/train", json={"retrain": True})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "metrics" in data
        assert "training_time" in data
    
    @patch('api.main.subprocess.run')
    def test_train_model_failure(self, mock_subprocess):
        """Test model training failure."""
        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Training failed"
        mock_subprocess.return_value = mock_result
        
        response = client.post("/train", json={"retrain": True})
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data
    
    @patch('api.main.Path.exists')
    @patch('api.main.load_metrics_json')
    def test_train_model_already_exists(self, mock_load_metrics, mock_exists):
        """Test training when model already exists."""
        # Mock model exists
        mock_exists.return_value = True
        
        # Mock metrics
        mock_metrics = {"accuracy": 0.85}
        mock_load_metrics.return_value = mock_metrics
        
        response = client.post("/train", json={"retrain": False})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "already trained" in data["message"]


class TestPrediction:
    """Test prediction endpoint."""
    
    @patch('api.main.ML_AVAILABLE', True)
    @patch('api.main.predict_and_explain')
    def test_predict_with_explanation(self, mock_predict):
        """Test prediction with SHAP explanation."""
        # Mock prediction result
        mock_result = {
            "probability": 0.75,
            "class": 1,
            "risk_level": "High",
            "confidence": 0.75,
            "explanations": [
                {"feature": "age", "shap_value": 0.3, "abs_shap_value": 0.3},
                {"feature": "cp", "shap_value": -0.2, "abs_shap_value": 0.2}
            ],
            "explanation_available": True
        }
        mock_predict.return_value = mock_result
        
        # Valid patient data
        patient_data = {
            "age": 67, "sex": 1, "cp": 3, "trestbps": 160, "chol": 286,
            "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
            "oldpeak": 4.1, "slope": 0, "ca": 3, "thal": 3
        }
        
        response = client.post("/predict", json=patient_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["probability"] == 0.75
        assert data["class_prediction"] == 1
        assert data["risk_level"] == "High"
        assert data["explanation_available"] is True
        assert len(data["top_features"]) == 2
    
    @patch('api.main.ML_AVAILABLE', True)
    @patch('api.main.predict_one')
    def test_predict_without_explanation(self, mock_predict):
        """Test prediction without SHAP explanation."""
        # Mock prediction result
        mock_result = {
            "probability": 0.3,
            "class": 0,
            "risk_level": "Low",
            "confidence": 0.7
        }
        mock_predict.return_value = mock_result
        
        patient_data = {
            "age": 35, "sex": 0, "cp": 0, "trestbps": 120, "chol": 180,
            "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
            "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
        }
        
        response = client.post(
            "/predict", 
            json=patient_data,
            params={"include_explanation": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["probability"] == 0.3
        assert data["class_prediction"] == 0
        assert data["risk_level"] == "Low"
        assert data["explanation_available"] is False
        assert data["top_features"] is None
    
    @patch('api.main.ML_AVAILABLE', False)
    def test_predict_ml_unavailable(self):
        """Test prediction fails when ML is unavailable."""
        patient_data = {
            "age": 50, "sex": 1, "cp": 1, "trestbps": 140, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }
        
        response = client.post("/predict", json=patient_data)
        
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        assert "ML inference not available" in data["detail"]
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid patient data."""
        invalid_data = {
            "age": 200,  # Invalid age
            "sex": 3,    # Invalid sex
            "cp": 5      # Invalid chest pain type
            # Missing other required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error


class TestMetrics:
    """Test metrics endpoint."""
    
    @patch('api.main.load_metrics_json')
    @patch('api.main.get_model_info')
    def test_get_metrics_available(self, mock_model_info, mock_load_metrics):
        """Test getting metrics when available."""
        # Mock metrics data
        mock_metrics = {
            "accuracy": 0.85,
            "roc_auc": 0.92,
            "f1_score": 0.83,
            "precision": 0.80,
            "recall": 0.87
        }
        mock_load_metrics.return_value = mock_metrics
        
        # Mock model info
        mock_model_info.return_value = {
            "model_exists": True,
            "metrics_exists": True,
            "metrics_modified": "2025-10-01T12:00:00"
        }
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metrics"] == mock_metrics
        assert data["model_info"]["model_exists"] is True
        assert data["last_updated"] is not None
    
    @patch('api.main.load_metrics_json')
    @patch('api.main.get_model_info')
    def test_get_metrics_unavailable(self, mock_model_info, mock_load_metrics):
        """Test getting metrics when not available."""
        # Mock no metrics
        mock_load_metrics.return_value = None
        
        # Mock model info
        mock_model_info.return_value = {
            "model_exists": False,
            "metrics_exists": False
        }
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["metrics"] is None
        assert data["model_info"]["model_exists"] is False


# Integration tests
class TestAPIIntegration:
    """Integration tests for API workflow."""
    
    def test_api_workflow_no_model(self):
        """Test API workflow when no model is trained."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Check status
        status_response = client.get("/status")
        assert status_response.status_code == 200
        
        # 3. Try prediction (should fail if ML not available)
        patient_data = {
            "age": 50, "sex": 1, "cp": 1, "trestbps": 140, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }
        
        predict_response = client.post("/predict", json=patient_data)
        # May succeed or fail depending on ML availability
        assert predict_response.status_code in [200, 503]
        
        # 4. Check metrics (should work even without model)
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])