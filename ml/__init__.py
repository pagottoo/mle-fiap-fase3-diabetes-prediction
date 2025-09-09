"""
Machine Learning module for Diabetes Health Indicators ML project.
Contains preprocessing, training, and inference utilities.
"""

from .preprocess import (
    build_preprocess_pipeline,
    load_raw_csv,
    split_stratified,
    preprocess_data,
    get_feature_names_after_transform,
    NUMERIC_COLUMNS,
    CATEGORICAL_ORDINAL_COLUMNS,
    BINARY_COLUMNS,
    ALL_FEATURE_COLUMNS
)

# Inference module imports
from .infer import (
    predict_one,
    explain_one,
    predict_and_explain,
    quick_predict,
    load_pipeline,
    ValidationError,
    InferenceError
)

# Training module will be imported when needed to avoid circular imports
# from .train import DiabetesTrainer

__version__ = "1.0.0"

__all__ = [
    # Preprocessing
    "build_preprocess_pipeline",
    "load_raw_csv", 
    "split_stratified",
    "preprocess_data",
    "get_feature_names_after_transform",
    "NUMERIC_COLUMNS",
    "CATEGORICAL_NOMINAL_COLUMNS", 
    "CATEGORICAL_ORDINAL_COLUMNS",
    "BINARY_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    # Inference
    "predict_one",
    "explain_one", 
    "predict_and_explain",
    "quick_predict",
    "load_pipeline",
    "ValidationError",
    "InferenceError"
]