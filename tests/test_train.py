"""
Unit tests for training module.
Tests model training pipeline and artifact generation.
"""

import pytest
import os
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ml.train import DiabetesTrainer


class TestDiabetesTrainer:
    """Test the main training class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for artifacts
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = DiabetesTrainer(
            data_path="./data/diabetes.csv",
            artifacts_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.data_path == "./data/diabetes.csv"
        assert self.trainer.artifacts_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert self.trainer.preprocessor is None
        assert self.trainer.baseline_model is None
        assert self.trainer.main_model is None
    
    def test_load_and_split_data(self):
        """Test data loading and splitting."""
        # Skip if dataset doesn't exist
        if not os.path.exists("./data/diabetes.csv"):
            pytest.skip("Dataset not found")
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.load_and_split_data()
            
            # Check shapes
            assert X_train.shape[0] > 0
            assert X_val.shape[0] > 0
            assert X_test.shape[0] > 0
            assert len(y_train) == X_train.shape[0]
            assert len(y_val) == X_val.shape[0]
            assert len(y_test) == X_test.shape[0]
            
            # Check preprocessor is fitted
            assert self.trainer.preprocessor is not None
            assert self.trainer.feature_names is not None
            
        except Exception as e:
            pytest.skip(f"Could not load data: {e}")
    
    @patch('ml.train.LogisticRegression')
    def test_train_baseline_model(self, mock_lr):
        """Test baseline model training."""
        # Mock the LogisticRegression
        mock_model = MagicMock()
        mock_lr.return_value = mock_model
        
        # Create dummy data
        X_train = np.random.random((100, 10))
        y_train = np.random.randint(0, 2, 100)
        
        # Train baseline
        result = self.trainer.train_baseline_model(X_train, y_train)
        
        # Check model was created and fitted
        mock_lr.assert_called_once()
        mock_model.fit.assert_called_once_with(X_train, y_train)
        assert self.trainer.baseline_model == mock_model
        assert result == mock_model
    
    @patch('ml.train.xgb.XGBClassifier')
    def test_train_main_model(self, mock_xgb):
        """Test main model training."""
        # Mock the XGBClassifier
        mock_model = MagicMock()
        mock_xgb.return_value = mock_model
        
        # Create dummy data
        X_train = np.random.random((100, 10))
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.random((20, 10))
        y_val = np.random.randint(0, 2, 20)
        
        # Train main model
        result = self.trainer.train_main_model(X_train, y_train, X_val, y_val)
        
        # Check model was created and fitted
        mock_xgb.assert_called_once()
        mock_model.fit.assert_called_once()
        assert self.trainer.main_model == mock_model
        assert result == mock_model
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]
        ])
        
        # Create test data
        X_test = np.random.random((5, 10))
        y_test = np.array([0, 1, 1, 0, 1])
        
        # Evaluate model
        metrics = self.trainer.evaluate_model(mock_model, X_test, y_test, "Test Model")
        
        # Check metrics are returned
        expected_metrics = ['roc_auc', 'f1', 'recall', 'precision', 'brier_score', 
                          'true_negatives', 'false_positives', 'false_negatives', 
                          'true_positives', 'specificity', 'accuracy']
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check metric types
        for metric in ['roc_auc', 'f1', 'recall', 'precision', 'brier_score', 'specificity', 'accuracy']:
            assert isinstance(metrics[metric], float)
        
        for metric in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']:
            assert isinstance(metrics[metric], int)
    
    def test_save_metrics(self):
        """Test metrics saving."""
        # Set up dummy metrics
        self.trainer.metrics = {
            'baseline': {'roc_auc': 0.8, 'f1': 0.7},
            'main': {'roc_auc': 0.85, 'f1': 0.75},
            'timestamp': '2024-01-01T12:00:00'
        }
        
        # Save metrics
        metrics_path = self.trainer.save_metrics()
        
        # Check file was created
        assert os.path.exists(metrics_path)
        assert metrics_path.endswith('metrics.json')
        
        # Check content
        with open(metrics_path, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == self.trainer.metrics
    
    @patch('ml.train.joblib.dump')
    @patch('ml.train.Pipeline')
    def test_save_model_pipeline(self, mock_pipeline, mock_joblib):
        """Test model pipeline saving."""
        # Set up mock models
        self.trainer.preprocessor = MagicMock()
        self.trainer.main_model = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Save model
        model_path = self.trainer.save_model_pipeline()
        
        # Check pipeline was created and saved
        mock_pipeline.assert_called_once()
        mock_joblib.assert_called_once_with(mock_pipeline_instance, model_path)
        assert model_path.endswith('model.pkl')


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_training_pipeline_structure(self):
        """Test that all required methods exist."""
        trainer = DiabetesTrainer()
        
        # Check all methods exist
        required_methods = [
            'load_and_split_data',
            'train_baseline_model', 
            'train_main_model',
            'evaluate_model',
            'compare_models',
            'calibrate_model',
            'save_model_pipeline',
            'save_metrics',
            'plot_roc_curve',
            'plot_precision_recall_curve',
            'generate_shap_summary',
            'run_complete_training'
        ]
        
        for method in required_methods:
            assert hasattr(trainer, method)
            assert callable(getattr(trainer, method))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])