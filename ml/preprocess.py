"""
Data preprocessing pipeline for Diabetes Health Indicators ML project.
Handles data loading, cleaning, feature engineering, and train/test splits.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Add project root to path to import database modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Feature column definitions - Diabetes Health Indicators (formato do Kaggle)
NUMERIC_COLUMNS = ['BMI', 'MentHlth', 'PhysHlth']
CATEGORICAL_ORDINAL_COLUMNS = ['GenHlth', 'Age', 'Education', 'Income']
BINARY_COLUMNS = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                  'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                  'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

# All feature columns (excluding target Diabetes_binary)
ALL_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + 
    CATEGORICAL_ORDINAL_COLUMNS + 
    BINARY_COLUMNS
)


def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Load diabetes dataset from CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with standardized column names
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Expected column names (Diabetes Health Indicators dataset standard)
    expected_columns = [
        'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
        'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
        'Sex', 'Age', 'Education', 'Income'
    ]
    
    # Check if we have the expected columns
    if not all(col in df.columns for col in expected_columns):
        # Try common alternative column names
        column_mapping = {
            'diabetes': 'Diabetes_binary',
            'target': 'Diabetes_binary',
            'class': 'Diabetes_binary',
            'diagnosis': 'Diabetes_binary'
        }
        
        # Apply mapping if needed
        df = df.rename(columns=column_mapping)
        
        # Check again after mapping
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only the columns we need
    df = df[expected_columns].copy()
    
    # Ensure target is binary: 0 (no diabetes) vs 1 (diabetes)
    if 'Diabetes_binary' in df.columns:
        df['Diabetes_binary'] = (df['Diabetes_binary'] > 0).astype(int)
    
    # Basic data validation and cleaning
    df = df.dropna(how='all')  # Remove completely empty rows
    df = df.drop_duplicates()  # Remove duplicate rows
    
    # Ensure proper data types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in CATEGORICAL_ORDINAL_COLUMNS + BINARY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    if 'Diabetes_binary' in df.columns:
        df['Diabetes_binary'] = pd.to_numeric(df['Diabetes_binary'], errors='coerce').astype('Int64')
    
    return df


def load_data_from_postgres() -> pd.DataFrame:
    """
    Load diabetes data from PostgreSQL database.
    
    Returns:
        pandas.DataFrame: Data loaded from diabetes_raw table
        
    Raises:
        ImportError: If database modules are not available
        Exception: If database connection or query fails
    """
    try:
        from api.db import SessionLocal
        from api.models import DiabetesRaw
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Query all records from diabetes_raw table
            records = db.query(DiabetesRaw).all()
            
            if not records:
                raise ValueError("No data found in database. Please upload data first via the Upload interface.")
            
            print(f"Loaded {len(records)} records from PostgreSQL database")
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'Diabetes_binary': record.diabetes_binary,
                    'HighBP': record.high_bp,
                    'HighChol': record.high_chol, 
                    'CholCheck': record.chol_check,
                    'BMI': record.bmi,
                    'Smoker': record.smoker,
                    'Stroke': record.stroke,
                    'HeartDiseaseorAttack': record.heart_disease_or_attack,
                    'PhysActivity': record.phys_activity,
                    'Fruits': record.fruits,
                    'Veggies': record.veggies,
                    'HvyAlcoholConsump': record.hvy_alcohol_consump,
                    'AnyHealthcare': record.any_healthcare,
                    'NoDocbcCost': record.no_docbc_cost,
                    'GenHlth': record.gen_hlth,
                    'MentHlth': record.ment_hlth,
                    'PhysHlth': record.phys_hlth,
                    'DiffWalk': record.diff_walking,
                    'Sex': record.sex,
                    'Age': record.age,
                    'Education': record.education,
                    'Income': record.income
                })
            
            df = pd.DataFrame(data)
            
            # Apply same cleaning as CSV loader
            df = df.dropna(how='all')
            df = df.drop_duplicates()
            
            # Ensure proper data types
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        finally:
            db.close()
            
    except ImportError as e:
        raise ImportError(f"Database modules not available: {e}")
    except Exception as e:
        raise Exception(f"Failed to load data from PostgreSQL: {e}")


def build_preprocess_pipeline() -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline with ColumnTransformer.
    
    Returns:
        sklearn.compose.ColumnTransformer: Fitted preprocessing pipeline
    """
    
    # Numeric features: median imputation + standard scaling
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical ordinal features: mode imputation + ordinal encoding
    # GenHlth: 1-5 (1=Excellent, 5=Poor)
    # Age: 1-13 (age categories)
    # Education: 1-6 (education levels)  
    # Income: 1-8 (income levels)
    categorical_ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Binary features: mode imputation (no scaling needed for 0/1)
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_COLUMNS),
            ('cat_ord', categorical_ordinal_pipeline, CATEGORICAL_ORDINAL_COLUMNS),
            ('binary', binary_pipeline, BINARY_COLUMNS)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def split_stratified(df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/validation/test sets with stratification.
    
    Args:
        df: Input dataframe with 'Diabetes_binary' column
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df) in 70/15/15 split
        
    Raises:
        ValueError: If 'Diabetes_binary' column is missing or has insufficient samples
    """
    if 'Diabetes_binary' not in df.columns:
        raise ValueError("DataFrame must contain 'Diabetes_binary' column for stratified split")
    
    if len(df) < 10:
        raise ValueError("Dataset too small for stratified split (minimum 10 samples required)")
    
    # Check if we have enough samples in each class
    target_counts = df['Diabetes_binary'].value_counts()
    if target_counts.min() < 3:
        raise ValueError("Each target class must have at least 3 samples for stratified split")
    
    # First split: 70% train, 30% temp (which will be split into 15% val, 15% test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=random_state,
        stratify=df['Diabetes_binary']
    )
    
    # Second split: 15% val, 15% test (50/50 split of the 30% temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df['Diabetes_binary']
    )
    
    return train_df, val_df, test_df


def get_feature_names_after_transform(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Get transformers
    transformers = preprocessor.transformers_
    
    for name, transformer, columns in transformers:
        if name == 'remainder':
            continue
            
        if isinstance(columns, str):
            columns = [columns]
        
        if name == 'num':
            # Numeric features keep original names
            feature_names.extend(columns)
        elif name == 'cat_nom':
            # One-hot encoded features
            if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                # For newer sklearn versions
                encoded_names = transformer.named_steps['encoder'].get_feature_names_out(columns)
                feature_names.extend(encoded_names)
            else:
                # Fallback for older versions
                categories = transformer.named_steps['encoder'].categories_
                for i, col in enumerate(columns):
                    for cat in categories[i]:
                        feature_names.append(f"{col}_{cat}")
        elif name == 'cat_ord':
            # Ordinal features keep original names
            feature_names.extend(columns)
        elif name == 'binary':
            # Binary features keep original names
            feature_names.extend(columns)
    
    return feature_names


def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Preprocess data using the preprocessing pipeline.
    
    Args:
        df: Input dataframe
        preprocessor: Existing preprocessor (if None, creates new one)
        fit: Whether to fit the preprocessor (True for training data)
        
    Returns:
        Tuple of (X_transformed, y, fitted_preprocessor)
    """
    # Separate features and target
    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df['Diabetes_binary'].values if 'Diabetes_binary' in df.columns else None
    
    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = build_preprocess_pipeline()
    
    # Fit and transform or just transform
    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)
    
    return X_transformed, y, preprocessor


if __name__ == "__main__":
    # Example usage
    try:
        # Load data
        df = load_raw_csv('./data/diabetes.csv')
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Target distribution:\n{df['Diabetes_binary'].value_counts()}")
        
        # Split data
        train_df, val_df, test_df = split_stratified(df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Build and test preprocessing pipeline
        preprocessor = build_preprocess_pipeline()
        X_train, y_train, fitted_preprocessor = preprocess_data(train_df, preprocessor, fit=True)
        
        print(f"Preprocessed training data shape: {X_train.shape}")
        print(f"Feature names: {get_feature_names_after_transform(fitted_preprocessor)}")
        
    except FileNotFoundError:
        print("Dataset not found. Please ensure data/diabetes.csv exists.")
    except Exception as e:
        print(f"Error: {e}")