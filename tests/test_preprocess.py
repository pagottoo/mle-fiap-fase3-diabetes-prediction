"""
Unit tests for preprocessing module.
Tests data loading, preprocessing pipeline, and train/test splits.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from ml.preprocess import (
    load_raw_csv,
    build_preprocess_pipeline,
    split_stratified,
    preprocess_data,
    get_feature_names_after_transform,
    NUMERIC_COLUMNS,
    CATEGORICAL_NOMINAL_COLUMNS,
    CATEGORICAL_ORDINAL_COLUMNS,
    BINARY_COLUMNS,
    ALL_FEATURE_COLUMNS
)


class TestConstants:
    """Test that constants are properly defined."""
    
    def test_column_constants(self):
        """Test that column constants are correctly defined."""
        assert isinstance(NUMERIC_COLUMNS, list)
        assert isinstance(CATEGORICAL_NOMINAL_COLUMNS, list)
        assert isinstance(CATEGORICAL_ORDINAL_COLUMNS, list)
        assert isinstance(BINARY_COLUMNS, list)
        assert isinstance(ALL_FEATURE_COLUMNS, list)
        
        # Check expected columns
        assert 'age' in NUMERIC_COLUMNS
        assert 'cp' in CATEGORICAL_NOMINAL_COLUMNS
        assert 'ca' in CATEGORICAL_ORDINAL_COLUMNS
        assert 'sex' in BINARY_COLUMNS
        
        # Check no overlap between categories
        all_lists = [NUMERIC_COLUMNS, CATEGORICAL_NOMINAL_COLUMNS, 
                    CATEGORICAL_ORDINAL_COLUMNS, BINARY_COLUMNS]
        all_cols = []
        for lst in all_lists:
            all_cols.extend(lst)
        
        assert len(all_cols) == len(set(all_cols)), "No overlap allowed between column categories"


class TestLoadRawCSV:
    """Test CSV loading functionality."""
    
    def create_sample_csv(self, filename):
        """Create a sample CSV file for testing."""
        data = {
            'age': [63, 37, 41, 56, 57],
            'sex': [1, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0],
            'trestbps': [145, 130, 130, 120, 120],
            'chol': [233, 250, 204, 236, 354],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 1],
            'thalach': [150, 187, 172, 178, 163],
            'exang': [0, 0, 0, 0, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
            'slope': [0, 0, 2, 2, 2],
            'ca': [0, 0, 0, 0, 0],
            'thal': [1, 2, 2, 2, 2],
            'target': [1, 1, 1, 1, 1]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return df
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df = self.create_sample_csv(f.name)
            
        try:
            df = load_raw_csv(f.name)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            assert 'target' in df.columns
            assert all(col in df.columns for col in ALL_FEATURE_COLUMNS)
        finally:
            os.unlink(f.name)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_raw_csv('nonexistent_file.csv')
    
    def test_target_binarization(self):
        """Test that target values are properly binarized."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = {
                'age': [63, 37, 41],
                'sex': [1, 1, 0],
                'cp': [3, 2, 1],
                'trestbps': [145, 130, 130],
                'chol': [233, 250, 204],
                'fbs': [1, 0, 0],
                'restecg': [0, 1, 0],
                'thalach': [150, 187, 172],
                'exang': [0, 0, 0],
                'oldpeak': [2.3, 3.5, 1.4],
                'slope': [0, 0, 2],
                'ca': [0, 0, 0],
                'thal': [1, 2, 2],
                'target': [0, 2, 4]  # Should be binarized to [0, 1, 1]
            }
            pd.DataFrame(data).to_csv(f.name, index=False)
        
        try:
            df = load_raw_csv(f.name)
            expected_target = [0, 1, 1]
            assert df['target'].tolist() == expected_target
        finally:
            os.unlink(f.name)


class TestBuildPreprocessPipeline:
    """Test preprocessing pipeline construction."""
    
    def test_pipeline_creation(self):
        """Test that pipeline is created correctly."""
        pipeline = build_preprocess_pipeline()
        assert isinstance(pipeline, ColumnTransformer)
        
        # Check that all transformers are present
        transformer_names = [name for name, _, _ in pipeline.transformers_]
        expected_names = ['num', 'cat_nom', 'cat_ord', 'binary']
        assert all(name in transformer_names for name in expected_names)
    
    def test_pipeline_components(self):
        """Test that pipeline components are correct types."""
        pipeline = build_preprocess_pipeline()
        
        for name, transformer, columns in pipeline.transformers_:
            if name == 'remainder':
                continue
                
            # Check that transformer is a Pipeline
            from sklearn.pipeline import Pipeline
            assert isinstance(transformer, Pipeline)
            
            # Check steps based on transformer type
            if name == 'num':
                assert 'imputer' in transformer.named_steps
                assert 'scaler' in transformer.named_steps
                assert isinstance(transformer.named_steps['scaler'], StandardScaler)
            elif name == 'cat_nom':
                assert 'imputer' in transformer.named_steps
                assert 'encoder' in transformer.named_steps
                assert isinstance(transformer.named_steps['encoder'], OneHotEncoder)
            elif name == 'cat_ord':
                assert 'imputer' in transformer.named_steps
                assert 'encoder' in transformer.named_steps
                assert isinstance(transformer.named_steps['encoder'], OrdinalEncoder)


class TestSplitStratified:
    """Test stratified data splitting."""
    
    def create_sample_dataframe(self, n_samples=100):
        """Create a sample dataframe for testing."""
        np.random.seed(42)
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(150, 400, n_samples),
            'fbs': np.random.choice([0, 1], n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.randint(100, 200, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.uniform(0, 5, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'ca': np.random.choice([0, 1, 2, 3], n_samples),
            'thal': np.random.choice([0, 1, 2, 3], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        return pd.DataFrame(data)
    
    def test_split_proportions(self):
        """Test that split proportions are approximately correct."""
        df = self.create_sample_dataframe(100)
        train_df, val_df, test_df = split_stratified(df)
        
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)
        
        # Allow some tolerance due to stratification
        assert 0.65 <= len(train_df) / total <= 0.75  # ~70%
        assert 0.10 <= len(val_df) / total <= 0.20    # ~15%
        assert 0.10 <= len(test_df) / total <= 0.20    # ~15%
    
    def test_split_stratification(self):
        """Test that splits maintain target distribution."""
        df = self.create_sample_dataframe(100)
        train_df, val_df, test_df = split_stratified(df)
        
        # Check that each split has both classes
        assert len(train_df['target'].unique()) == 2
        assert len(val_df['target'].unique()) == 2
        assert len(test_df['target'].unique()) == 2
    
    def test_split_no_target_column(self):
        """Test that split fails when target column is missing."""
        df = self.create_sample_dataframe(100)
        df = df.drop('target', axis=1)
        
        with pytest.raises(ValueError, match="DataFrame must contain 'target' column"):
            split_stratified(df)
    
    def test_split_insufficient_samples(self):
        """Test that split fails with insufficient samples."""
        df = self.create_sample_dataframe(5)  # Too few samples
        
        with pytest.raises(ValueError, match="Dataset too small"):
            split_stratified(df)


class TestPreprocessData:
    """Test data preprocessing functionality."""
    
    def create_sample_dataframe(self):
        """Create a sample dataframe for testing."""
        data = {
            'age': [63, 37, 41, 56, 57],
            'sex': [1, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0],
            'trestbps': [145, 130, 130, 120, 120],
            'chol': [233, 250, 204, 236, 354],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 1],
            'thalach': [150, 187, 172, 178, 163],
            'exang': [0, 0, 0, 0, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
            'slope': [0, 0, 2, 2, 2],
            'ca': [0, 0, 0, 0, 0],
            'thal': [1, 2, 2, 2, 2],
            'target': [1, 1, 1, 1, 1]
        }
        return pd.DataFrame(data)
    
    def test_preprocess_data_fit(self):
        """Test preprocessing with fit=True."""
        df = self.create_sample_dataframe()
        X_transformed, y, preprocessor = preprocess_data(df, fit=True)
        
        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(preprocessor, ColumnTransformer)
        assert X_transformed.shape[0] == len(df)
        assert len(y) == len(df)
    
    def test_preprocess_data_transform_only(self):
        """Test preprocessing with fit=False."""
        df = self.create_sample_dataframe()
        
        # First fit
        _, _, fitted_preprocessor = preprocess_data(df, fit=True)
        
        # Then transform only
        X_transformed, y, preprocessor = preprocess_data(df, fitted_preprocessor, fit=False)
        
        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert preprocessor is fitted_preprocessor


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])