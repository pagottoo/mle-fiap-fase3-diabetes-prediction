"""
Inference module for Diabetes Health Indicators ML project.
Handles model loading, prediction, and SHAP explanations.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Union, Optional
import warnings
import logging

# SHAP for explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Explanations will not work.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected feature order for the model - Diabetes Health Indicators
EXPECTED_FEATURES = [
    'BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income',
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'DiffWalk', 'Sex'
]

# Global variable to cache loaded pipeline
_cached_pipeline = None
_cached_explainer = None


class InferenceError(Exception):
    """Custom exception for inference errors."""
    pass


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input payload for prediction.
    
    Args:
        payload: Input features dictionary
        
    Returns:
        Validated and cleaned payload
        
    Raises:
        ValidationError: If payload is invalid
    """
    if not isinstance(payload, dict):
        raise ValidationError("Payload must be a dictionary")
    
    # Check if all required features are present
    missing_features = set(EXPECTED_FEATURES) - set(payload.keys())
    if missing_features:
        raise ValidationError(f"Missing required features: {missing_features}")
    
    # Check for extra features (warn but don't fail)
    extra_features = set(payload.keys()) - set(EXPECTED_FEATURES)
    if extra_features:
        logger.warning(f"Extra features will be ignored: {extra_features}")
    
    # Validate feature types and ranges for Diabetes Health Indicators
    validated_payload = {}
    
    # Numeric features
    bmi = payload.get('BMI')
    if not isinstance(bmi, (int, float)) or not (10 <= bmi <= 100):
        raise ValidationError(f"BMI must be between 10-100, got: {bmi}")
    validated_payload['BMI'] = float(bmi)
    
    ment_hlth = payload.get('MentHlth')
    if not isinstance(ment_hlth, (int, float)) or not (0 <= ment_hlth <= 30):
        raise ValidationError(f"MentHlth must be between 0-30, got: {ment_hlth}")
    validated_payload['MentHlth'] = float(ment_hlth)
    
    phys_hlth = payload.get('PhysHlth')
    if not isinstance(phys_hlth, (int, float)) or not (0 <= phys_hlth <= 30):
        raise ValidationError(f"PhysHlth must be between 0-30, got: {phys_hlth}")
    validated_payload['PhysHlth'] = float(phys_hlth)
    
    # Ordinal categorical features
    gen_hlth = payload.get('GenHlth')
    if gen_hlth not in [1, 2, 3, 4, 5]:
        raise ValidationError(f"GenHlth must be 1-5, got: {gen_hlth}")
    validated_payload['GenHlth'] = int(gen_hlth)
    
    age = payload.get('Age')
    if age not in list(range(1, 14)):
        raise ValidationError(f"Age category must be 1-13, got: {age}")
    validated_payload['Age'] = int(age)
    
    education = payload.get('Education')
    if education not in list(range(1, 7)):
        raise ValidationError(f"Education must be 1-6, got: {education}")
    validated_payload['Education'] = int(education)
    
    income = payload.get('Income')
    if income not in list(range(1, 9)):
        raise ValidationError(f"Income must be 1-8, got: {income}")
    validated_payload['Income'] = int(income)
    
    # Binary features (0 or 1)
    binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
    
    for feature in binary_features:
        value = payload.get(feature)
        if value not in [0, 1]:
            raise ValidationError(f"{feature} must be 0 or 1, got: {value}")
        validated_payload[feature] = int(value)
    
    return validated_payload


def load_pipeline(model_path: str = "./ml/artifacts/model.pkl") -> object:
    """
    Load the trained ML pipeline from disk.
    
    Args:
        model_path: Path to the saved model pipeline
        
    Returns:
        Loaded sklearn pipeline
        
    Raises:
        InferenceError: If model cannot be loaded
    """
    global _cached_pipeline
    
    # Return cached pipeline if available
    if _cached_pipeline is not None:
        logger.info("Using cached pipeline")
        return _cached_pipeline
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise InferenceError(f"Model file not found: {model_path}")
    
    try:
        # Load pipeline
        pipeline = joblib.load(model_path)
        
        # Validate pipeline structure
        if not hasattr(pipeline, 'predict') or not hasattr(pipeline, 'predict_proba'):
            raise InferenceError("Loaded object is not a valid ML pipeline")
        
        # Cache for future use
        _cached_pipeline = pipeline
        
        logger.info(f"Pipeline loaded successfully from: {model_path}")
        return pipeline
        
    except Exception as e:
        raise InferenceError(f"Failed to load pipeline: {str(e)}")


def _create_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Create DataFrame from payload with features in correct order.
    
    Args:
        payload: Validated feature dictionary
        
    Returns:
        Single-row DataFrame with features in expected order
    """
    # Create DataFrame with single row and features in correct order
    data = {feature: [payload[feature]] for feature in EXPECTED_FEATURES}
    df = pd.DataFrame(data)
    
    return df


def predict_one(payload: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Make prediction for a single patient.
    
    Args:
        payload: Dictionary with patient features
        threshold: Classification threshold (default 0.5)
        
    Returns:
        Dictionary with prediction results
        
    Raises:
        ValidationError: If payload is invalid
        InferenceError: If prediction fails
    """
    try:
        # Validate input
        validated_payload = validate_payload(payload)
        
        # Load pipeline
        pipeline = load_pipeline()
        
        # Create DataFrame
        df = _create_dataframe(validated_payload)
        
        # Make predictions
        prediction_proba = pipeline.predict_proba(df)[0]  # Get first (and only) row
        probability = float(prediction_proba[1])  # Probability of positive class
        
        # Apply threshold
        prediction_class = int(probability >= threshold)
        
        # Create result
        result = {
            'probability': probability,
            'class': prediction_class,
            'threshold': threshold,
            'confidence': max(prediction_proba),  # Confidence is max probability
            'risk_level': _get_risk_level(probability)
        }
        
        logger.info(f"Prediction completed: class={prediction_class}, probability={probability:.3f}")
        return result
        
    except (ValidationError, InferenceError):
        raise
    except Exception as e:
        raise InferenceError(f"Prediction failed: {str(e)}")


def _get_risk_level(probability: float) -> str:
    """
    Convert probability to risk level category.
    
    Args:
        probability: Probability of diabetes
        
    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def _get_explainer(pipeline) -> Optional[object]:
    """
    Get or create SHAP explainer for the pipeline.
    
    Args:
        pipeline: Trained ML pipeline
        
    Returns:
        SHAP explainer or None if not available
    """
    global _cached_explainer
    
    if not SHAP_AVAILABLE:
        return None
    
    if _cached_explainer is not None:
        return _cached_explainer
    
    try:
        # Get the model from pipeline (last step)
        model = pipeline.named_steps['classifier']
        
        # Create appropriate explainer based on model type
        if hasattr(model, 'tree_'):
            # Single tree models (unlikely)
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'estimators_') or 'XGB' in str(type(model)):
            # Tree-based ensemble models (XGBoost, RandomForest, etc.)
            explainer = shap.TreeExplainer(model)
        else:
            # Linear models or others - use more general explainer
            # For this we'd need background data, so we'll skip for now
            logger.warning("Model type not supported for SHAP explanations")
            return None
        
        _cached_explainer = explainer
        return explainer
        
    except Exception as e:
        logger.warning(f"Could not create SHAP explainer: {e}")
        return None


def explain_one(payload: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Generate SHAP explanation for a single prediction.
    
    Args:
        payload: Dictionary with patient features
        top_k: Number of top features to return
        
    Returns:
        List of feature explanations, sorted by absolute importance
        
    Raises:
        ValidationError: If payload is invalid
        InferenceError: If explanation fails
    """
    if not SHAP_AVAILABLE:
        raise InferenceError("SHAP not available. Cannot generate explanations.")
    
    try:
        # Validate input
        validated_payload = validate_payload(payload)
        
        # Load pipeline
        pipeline = load_pipeline()
        
        # Get explainer
        explainer = _get_explainer(pipeline)
        if explainer is None:
            raise InferenceError("Could not create SHAP explainer for this model")
        
        # Create DataFrame and preprocess
        df = _create_dataframe(validated_payload)
        
        # Transform data using only the preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(df)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class case - take positive class (index 1)
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            # Binary case or single output
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Get feature names after preprocessing
        feature_names = _get_feature_names_after_preprocessing(preprocessor)
        
        # Create feature-value pairs
        if len(values) != len(feature_names):
            logger.warning(f"Shape mismatch: {len(values)} SHAP values vs {len(feature_names)} features")
            # Fall back to simple feature names
            feature_names = [f"feature_{i}" for i in range(len(values))]
        
        explanations = []
        for i, (name, shap_val) in enumerate(zip(feature_names, values)):
            explanations.append({
                'feature': name,
                'shap_value': float(shap_val),
                'abs_shap_value': abs(float(shap_val))
            })
        
        # Sort by absolute SHAP value (descending)
        explanations.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        # Return top_k explanations
        top_explanations = explanations[:top_k]
        
        logger.info(f"Generated explanations for top {len(top_explanations)} features")
        return top_explanations
        
    except (ValidationError, InferenceError):
        raise
    except Exception as e:
        raise InferenceError(f"Explanation failed: {str(e)}")


def _get_feature_names_after_preprocessing(preprocessor) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    
    Args:
        preprocessor: Fitted preprocessing pipeline
        
    Returns:
        List of feature names
    """
    try:
        # Try to get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out())
        
        # Fallback: construct names based on transformers
        from ml.preprocess import get_feature_names_after_transform
        return get_feature_names_after_transform(preprocessor)
        
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        # Final fallback
        return [f"feature_{i}" for i in range(len(EXPECTED_FEATURES))]


def predict_and_explain(payload: Dict[str, Any], threshold: float = 0.5, 
                       top_k: int = 3) -> Dict[str, Any]:
    """
    Combine prediction and explanation in a single call.
    
    Args:
        payload: Dictionary with patient features
        threshold: Classification threshold
        top_k: Number of top SHAP features to return
        
    Returns:
        Combined prediction and explanation results
    """
    try:
        # Get prediction
        prediction_result = predict_one(payload, threshold)
        
        # Get explanation (if available)
        explanation_result = []
        try:
            explanation_result = explain_one(payload, top_k)
        except InferenceError as e:
            logger.warning(f"Explanation failed: {e}")
        
        # Combine results
        result = {
            **prediction_result,
            'explanations': explanation_result,
            'explanation_available': len(explanation_result) > 0
        }
        
        return result
        
    except Exception as e:
        raise InferenceError(f"Combined prediction and explanation failed: {str(e)}")


# Convenience functions for common use cases
def quick_predict(age: float, sex: int, cp: int, trestbps: float, chol: float,
                 fbs: int, restecg: int, thalach: float, exang: int, 
                 oldpeak: float, slope: int, ca: int, thal: int) -> Dict[str, Any]:
    """
    Quick prediction with individual parameters instead of dictionary.
    
    Returns:
        Prediction result dictionary
    """
    payload = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    return predict_one(payload)


if __name__ == "__main__":
    # Example usage and testing
    print("Diabetes Health Indicators Inference Module")
    print("=" * 40)
    
    # Sample patient data
    sample_payload = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    
    try:
        # Test pipeline loading
        pipeline = load_pipeline()
        print("Pipeline loaded successfully")
        
        # Test prediction
        result = predict_one(sample_payload)
        print(f"Prediction: class={result['class']}, probability={result['probability']:.3f}")

        # Test explanation
        if SHAP_AVAILABLE:
            explanations = explain_one(sample_payload)
            print(f"Explanations generated: {len(explanations)} features")
            for exp in explanations:
                print(f"   {exp['feature']}: {exp['shap_value']:.3f}")
        else:
            print("SHAP not available for explanations")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()