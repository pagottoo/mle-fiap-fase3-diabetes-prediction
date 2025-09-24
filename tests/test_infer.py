"""
Unit tests for the inference module.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from unittest.mock import patch, MagicMock

# Import the inference module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.infer import (
    validate_payload, load_pipeline, predict_one, explain_one,
    predict_and_explain, quick_predict, ValidationError, InferenceError,
    EXPECTED_FEATURES, _create_dataframe, _get_risk_level
)


class TestValidatePayload(unittest.TestCase):
    """Test payload validation."""
    
    def setUp(self):
        """Set up valid payload for testing."""
        self.valid_payload = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
    
    def test_valid_payload(self):
        """Test that valid payload passes validation."""
        result = validate_payload(self.valid_payload)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(EXPECTED_FEATURES))
        
        # Check types
        self.assertIsInstance(result['age'], float)
        self.assertIsInstance(result['sex'], int)
        self.assertIsInstance(result['cp'], int)
    
    def test_missing_features(self):
        """Test validation fails when features are missing."""
        incomplete_payload = {'age': 63, 'sex': 1}  # Missing other features
        
        with self.assertRaises(ValidationError) as context:
            validate_payload(incomplete_payload)
        
        self.assertIn("Missing required features", str(context.exception))
    
    def test_invalid_age(self):
        """Test validation fails for invalid age."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['age'] = 150  # Too old
        
        with self.assertRaises(ValidationError) as context:
            validate_payload(invalid_payload)
        
        self.assertIn("Age must be a number between 1 and 120", str(context.exception))
    
    def test_invalid_sex(self):
        """Test validation fails for invalid sex."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['sex'] = 2  # Invalid value
        
        with self.assertRaises(ValidationError) as context:
            validate_payload(invalid_payload)
        
        self.assertIn("Sex must be 0 (female) or 1 (male)", str(context.exception))
    
    def test_invalid_chest_pain(self):
        """Test validation fails for invalid chest pain type."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['cp'] = 5  # Invalid value
        
        with self.assertRaises(ValidationError) as context:
            validate_payload(invalid_payload)
        
        self.assertIn("Chest pain type (cp) must be 0-3", str(context.exception))
    
    def test_invalid_blood_pressure(self):
        """Test validation fails for invalid blood pressure."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['trestbps'] = 50  # Too low
        
        with self.assertRaises(ValidationError) as context:
            validate_payload(invalid_payload)
        
        self.assertIn("Resting blood pressure must be between 80-250", str(context.exception))
    
    def test_extra_features_warning(self):
        """Test that extra features generate warning but don't fail."""
        payload_with_extra = self.valid_payload.copy()
        payload_with_extra['extra_feature'] = 123
        
        # Should not raise exception
        result = validate_payload(payload_with_extra)
        self.assertEqual(len(result), len(EXPECTED_FEATURES))
    
    def test_non_dict_payload(self):
        """Test validation fails for non-dictionary input."""
        with self.assertRaises(ValidationError) as context:
            validate_payload("not a dict")
        
        self.assertIn("Payload must be a dictionary", str(context.exception))


class TestCreateDataFrame(unittest.TestCase):
    """Test DataFrame creation from payload."""
    
    def test_create_dataframe(self):
        """Test DataFrame creation with correct column order."""
        payload = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
        
        df = _create_dataframe(payload)
        
        # Check shape
        self.assertEqual(df.shape, (1, len(EXPECTED_FEATURES)))
        
        # Check column order
        self.assertEqual(list(df.columns), EXPECTED_FEATURES)
        
        # Check values
        self.assertEqual(df.iloc[0]['age'], 63)
        self.assertEqual(df.iloc[0]['sex'], 1)


class TestRiskLevel(unittest.TestCase):
    """Test risk level categorization."""
    
    def test_low_risk(self):
        """Test low risk categorization."""
        self.assertEqual(_get_risk_level(0.1), "Low")
        self.assertEqual(_get_risk_level(0.29), "Low")
    
    def test_medium_risk(self):
        """Test medium risk categorization."""
        self.assertEqual(_get_risk_level(0.3), "Medium")
        self.assertEqual(_get_risk_level(0.5), "Medium")
        self.assertEqual(_get_risk_level(0.69), "Medium")
    
    def test_high_risk(self):
        """Test high risk categorization."""
        self.assertEqual(_get_risk_level(0.7), "High")
        self.assertEqual(_get_risk_level(0.9), "High")


class TestLoadPipeline(unittest.TestCase):
    """Test pipeline loading functionality."""
    
    def test_load_nonexistent_file(self):
        """Test loading fails when file doesn't exist."""
        with self.assertRaises(InferenceError) as context:
            load_pipeline("nonexistent/path/model.pkl")
        
        self.assertIn("Model file not found", str(context.exception))
    
    def test_load_invalid_file(self):
        """Test loading fails when file is not a valid pipeline."""
        # Create temporary file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            joblib.dump("not a pipeline", temp_file.name)
            temp_path = temp_file.name
        
        try:
            with self.assertRaises(InferenceError) as context:
                load_pipeline(temp_path)
            
            self.assertIn("not a valid ML pipeline", str(context.exception))
        finally:
            os.unlink(temp_path)
    
    @patch('ml.infer._cached_pipeline', None)
    def test_successful_load(self):
        """Test successful pipeline loading."""
        # Create mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict = MagicMock()
        mock_pipeline.predict_proba = MagicMock()
        
        # Create temporary file with mock pipeline
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            joblib.dump(mock_pipeline, temp_file.name)
            temp_path = temp_file.name
        
        try:
            pipeline = load_pipeline(temp_path)
            
            self.assertIsNotNone(pipeline)
            self.assertTrue(hasattr(pipeline, 'predict'))
            self.assertTrue(hasattr(pipeline, 'predict_proba'))
        finally:
            os.unlink(temp_path)


class TestPredictOne(unittest.TestCase):
    """Test single prediction functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_payload = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
    
    @patch('ml.infer.load_pipeline')
    def test_predict_one_success(self):
        """Test successful prediction."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])  # [neg_prob, pos_prob]
        
        load_pipeline_mock = mock_pipeline
        
        with patch('ml.infer.load_pipeline', return_value=mock_pipeline):
            result = predict_one(self.valid_payload, threshold=0.5)
        
        # Check result structure
        self.assertIn('probability', result)
        self.assertIn('class', result)
        self.assertIn('threshold', result)
        self.assertIn('confidence', result)
        self.assertIn('risk_level', result)
        
        # Check values
        self.assertEqual(result['probability'], 0.7)
        self.assertEqual(result['class'], 1)  # 0.7 >= 0.5
        self.assertEqual(result['threshold'], 0.5)
        self.assertEqual(result['confidence'], 0.7)  # max probability
        self.assertEqual(result['risk_level'], "High")
    
    @patch('ml.infer.load_pipeline')
    def test_predict_one_with_custom_threshold(self):
        """Test prediction with custom threshold."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.6, 0.4]])
        
        with patch('ml.infer.load_pipeline', return_value=mock_pipeline):
            result = predict_one(self.valid_payload, threshold=0.3)
        
        self.assertEqual(result['probability'], 0.4)
        self.assertEqual(result['class'], 1)  # 0.4 >= 0.3
        self.assertEqual(result['threshold'], 0.3)
    
    def test_predict_one_invalid_payload(self):
        """Test prediction fails with invalid payload."""
        invalid_payload = {'age': 63}  # Missing features
        
        with self.assertRaises(ValidationError):
            predict_one(invalid_payload)


class TestQuickPredict(unittest.TestCase):
    """Test quick predict convenience function."""
    
    @patch('ml.infer.predict_one')
    def test_quick_predict(self):
        """Test quick predict function."""
        # Mock predict_one
        expected_result = {'probability': 0.6, 'class': 1}
        
        with patch('ml.infer.predict_one', return_value=expected_result) as mock_predict:
            result = quick_predict(
                age=63, sex=1, cp=3, trestbps=145, chol=233,
                fbs=1, restecg=0, thalach=150, exang=0,
                oldpeak=2.3, slope=0, ca=0, thal=1
            )
        
        # Check that predict_one was called with correct payload
        mock_predict.assert_called_once()
        args, kwargs = mock_predict.call_args
        payload = args[0]
        
        self.assertEqual(payload['age'], 63)
        self.assertEqual(payload['sex'], 1)
        self.assertEqual(len(payload), 13)
        
        # Check result
        self.assertEqual(result, expected_result)


class TestExplainOne(unittest.TestCase):
    """Test SHAP explanation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_payload = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
    
    @patch('ml.infer.SHAP_AVAILABLE', False)
    def test_explain_one_shap_unavailable(self):
        """Test explanation fails when SHAP is unavailable."""
        with self.assertRaises(InferenceError) as context:
            explain_one(self.valid_payload)
        
        self.assertIn("SHAP not available", str(context.exception))
    
    @patch('ml.infer.SHAP_AVAILABLE', True)
    @patch('ml.infer._get_explainer')
    @patch('ml.infer.load_pipeline')
    def test_explain_one_no_explainer(self, mock_load, mock_get_explainer):
        """Test explanation fails when explainer cannot be created."""
        mock_get_explainer.return_value = None
        
        with self.assertRaises(InferenceError) as context:
            explain_one(self.valid_payload)
        
        self.assertIn("Could not create SHAP explainer", str(context.exception))
    
    @patch('ml.infer.SHAP_AVAILABLE', True)
    @patch('ml.infer._get_explainer')
    @patch('ml.infer.load_pipeline')
    @patch('ml.infer._get_feature_names_after_preprocessing')
    def test_explain_one_success(self, mock_features, mock_load, mock_get_explainer):
        """Test successful explanation generation."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        mock_pipeline.named_steps = {'preprocessor': mock_preprocessor}
        mock_load.return_value = mock_pipeline
        
        # Mock explainer
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.3, 0.5, -0.2, 0.4]])
        mock_get_explainer.return_value = mock_explainer
        
        # Mock feature names
        mock_features.return_value = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        
        result = explain_one(self.valid_payload, top_k=3)
        
        # Check result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # top_k
        
        # Check explanation structure
        for exp in result:
            self.assertIn('feature', exp)
            self.assertIn('shap_value', exp)
            self.assertIn('abs_shap_value', exp)
        
        # Check ordering (by absolute value, descending)
        abs_values = [exp['abs_shap_value'] for exp in result]
        self.assertEqual(abs_values, sorted(abs_values, reverse=True))


class TestPredictAndExplain(unittest.TestCase):
    """Test combined prediction and explanation."""
    
    @patch('ml.infer.predict_one')
    @patch('ml.infer.explain_one')
    def test_predict_and_explain_success(self, mock_explain, mock_predict):
        """Test successful combined prediction and explanation."""
        # Mock results
        mock_predict.return_value = {'probability': 0.7, 'class': 1}
        mock_explain.return_value = [
            {'feature': 'age', 'shap_value': 0.3, 'abs_shap_value': 0.3}
        ]
        
        payload = {'age': 63}  # Simplified for test
        result = predict_and_explain(payload)
        
        # Check combined result
        self.assertIn('probability', result)
        self.assertIn('class', result)
        self.assertIn('explanations', result)
        self.assertIn('explanation_available', result)
        
        self.assertEqual(result['probability'], 0.7)
        self.assertEqual(result['class'], 1)
        self.assertEqual(len(result['explanations']), 1)
        self.assertTrue(result['explanation_available'])
    
    @patch('ml.infer.predict_one')
    @patch('ml.infer.explain_one')
    def test_predict_and_explain_explanation_fails(self, mock_explain, mock_predict):
        """Test combined function when explanation fails."""
        # Mock results
        mock_predict.return_value = {'probability': 0.7, 'class': 1}
        mock_explain.side_effect = InferenceError("SHAP failed")
        
        payload = {'age': 63}
        result = predict_and_explain(payload)
        
        # Should still have prediction results
        self.assertEqual(result['probability'], 0.7)
        self.assertEqual(result['class'], 1)
        
        # But no explanations
        self.assertEqual(result['explanations'], [])
        self.assertFalse(result['explanation_available'])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)