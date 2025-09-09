"""
Training script for Diabetes Health Indicators ML project.
Implements baseline and XGBoost models with evaluation and persistence.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from datetime import datetime
import joblib
import warnings

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

# Local imports
from ml.preprocess import (
    load_raw_csv, build_preprocess_pipeline, split_stratified,
    preprocess_data, get_feature_names_after_transform
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")


class DiabetesTrainer:
    """Main trainer class for diabetes risk models."""
    
    def __init__(self, data_path: str = "./data/diabetes.csv", artifacts_dir: str = "./ml/artifacts"):
        """
        Initialize trainer with data path and artifacts directory.
        
        Args:
            data_path: Path to the diabetes dataset
            artifacts_dir: Directory to save model artifacts
        """
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.preprocessor = None
        self.baseline_model = None
        self.main_model = None
        self.calibrated_model = None
        self.feature_names = None
        self.metrics = {}
        self.data_info = {}  # Store information about training data
        
        # Create artifacts directory
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def load_and_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data and create train/validation/test splits."""
        print("Loading and splitting data...")
        
        # Try to load data from PostgreSQL first, fallback to CSV
        try:
            from ml.preprocess import load_data_from_postgres
            df = load_data_from_postgres()
            print(f"Loaded dataset from PostgreSQL with shape: {df.shape}")
            self.data_info['source'] = 'PostgreSQL database'
            self.data_info['total_records'] = len(df)
        except Exception as e:
            print(f"Could not load from PostgreSQL ({e}), falling back to CSV file")
            df = load_raw_csv(self.data_path)
            print(f"Loaded dataset from CSV file with shape: {df.shape}")
            self.data_info['source'] = f'CSV file ({self.data_path})'
            self.data_info['total_records'] = len(df)
        
        print(f"Target distribution:\n{df['Diabetes_binary'].value_counts()}")
        
        # Split data
        train_df, val_df, test_df = split_stratified(df, random_state=42)
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Store split information
        self.data_info['train_records'] = len(train_df)
        self.data_info['validation_records'] = len(val_df)
        self.data_info['test_records'] = len(test_df)
        
        # Preprocess data
        self.preprocessor = build_preprocess_pipeline()
        
        # Fit on training data
        X_train, y_train, fitted_preprocessor = preprocess_data(train_df, self.preprocessor, fit=True)
        self.preprocessor = fitted_preprocessor
        
        # Transform validation and test data
        X_val, y_val, _ = preprocess_data(val_df, self.preprocessor, fit=False)
        X_test, y_test, _ = preprocess_data(test_df, self.preprocessor, fit=False)
        
        # Get feature names
        self.feature_names = get_feature_names_after_transform(self.preprocessor)
        
        print(f"Preprocessed shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Number of features after preprocessing: {X_train.shape[1]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_baseline_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train baseline Logistic Regression model."""
        print("\nTraining baseline model (Logistic Regression)...")
        
        baseline = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )
        
        baseline.fit(X_train, y_train)
        self.baseline_model = baseline
        
        print("Baseline model training completed")
        return baseline
    
    def train_main_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        """Train main XGBoost model."""
        print("\nTraining main model (XGBoost)...")
        
        main_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False
        )
        
        # Train with early stopping on validation set
        main_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.main_model = main_model
        print("Main model training completed")
        return main_model
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        })
        
        print(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return metrics
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> bool:
        """Compare baseline and main model performance."""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Evaluate baseline
        baseline_metrics = self.evaluate_model(self.baseline_model, X_test, y_test, "Baseline (Logistic Regression)")
        
        # Evaluate main model
        main_metrics = self.evaluate_model(self.main_model, X_test, y_test, "Main (XGBoost)")
        
        # Store metrics
        self.metrics = {
            'baseline': baseline_metrics,
            'main': main_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Compare AUC scores
        baseline_auc = baseline_metrics['roc_auc']
        main_auc = main_metrics['roc_auc']
        
        print(f"\nAUC Comparison:")
        print(f"  Baseline: {baseline_auc:.4f}")
        print(f"  Main:     {main_auc:.4f}")
        print(f"  Improvement: {main_auc - baseline_auc:.4f}")
        
        is_better = main_auc >= baseline_auc
        print(f"\nMain model is {'BETTER' if is_better else 'WORSE'} than baseline")
        
        return is_better
    
    def calibrate_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> CalibratedClassifierCV:
        """Calibrate model if Brier score is poor."""
        main_brier = self.metrics['main']['brier_score']
        
        print(f"\nChecking calibration (Brier score: {main_brier:.4f})...")
        
        # If Brier score > 0.25, apply calibration
        if main_brier > 0.25:
            print("Applying isotonic calibration...")
            
            calibrated = CalibratedClassifierCV(
                self.main_model, 
                method='isotonic', 
                cv=3
            )
            
            calibrated.fit(X_train, y_train)
            self.calibrated_model = calibrated
            
            # Evaluate calibrated model
            calibrated_metrics = self.evaluate_model(calibrated, X_test, y_test, "Calibrated XGBoost")
            self.metrics['calibrated'] = calibrated_metrics
            
            print(f"Calibration improved Brier score: {main_brier:.4f} -> {calibrated_metrics['brier_score']:.4f}")
            return calibrated
        else:
            print("Calibration not needed (Brier score is acceptable)")
            return None
    
    def save_model_pipeline(self) -> str:
        """Save complete preprocessing + model pipeline."""
        print("\nSaving model pipeline...")
        
        # Choose best model (calibrated if available, otherwise main)
        final_model = self.calibrated_model if self.calibrated_model else self.main_model
        
        # Create complete pipeline
        complete_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', final_model)
        ])
        
        # Save pipeline
        model_path = os.path.join(self.artifacts_dir, 'model.pkl')
        joblib.dump(complete_pipeline, model_path)
        
        print(f"Model pipeline saved to: {model_path}")
        return model_path
    
    def save_metrics(self) -> str:
        """Save evaluation metrics to JSON."""
        metrics_path = os.path.join(self.artifacts_dir, 'metrics.json')
        
        # Include data information in metrics
        full_metrics = {
            'data_info': self.data_info,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2, default=str)
        
        print(f"Metrics saved to: {metrics_path}")
        return metrics_path
    
    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Generate and save ROC curve plot."""
        print("Generating ROC curve...")
        
        plt.figure(figsize=(10, 8))
        
        # Baseline ROC
        y_pred_proba_baseline = self.baseline_model.predict_proba(X_test)[:, 1]
        fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
        baseline_auc = self.metrics['baseline']['roc_auc']
        
        # Main model ROC
        y_pred_proba_main = self.main_model.predict_proba(X_test)[:, 1]
        fpr_main, tpr_main, _ = roc_curve(y_test, y_pred_proba_main)
        main_auc = self.metrics['main']['roc_auc']
        
        # Plot curves
        plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC = {baseline_auc:.3f})', linewidth=2)
        plt.plot(fpr_main, tpr_main, label=f'XGBoost (AUC = {main_auc:.3f})', linewidth=2)
        
        # Calibrated model if available
        if self.calibrated_model:
            y_pred_proba_cal = self.calibrated_model.predict_proba(X_test)[:, 1]
            fpr_cal, tpr_cal, _ = roc_curve(y_test, y_pred_proba_cal)
            cal_auc = self.metrics['calibrated']['roc_auc']
            plt.plot(fpr_cal, tpr_cal, label=f'Calibrated XGBoost (AUC = {cal_auc:.3f})', linewidth=2)
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Diabetes Risk Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.artifacts_dir, 'roc.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to: {roc_path}")
        return roc_path
    
    def plot_precision_recall_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Generate and save Precision-Recall curve plot."""
        print("Generating Precision-Recall curve...")
        
        plt.figure(figsize=(10, 8))
        
        # Baseline PR
        y_pred_proba_baseline = self.baseline_model.predict_proba(X_test)[:, 1]
        precision_baseline, recall_baseline, _ = precision_recall_curve(y_test, y_pred_proba_baseline)
        
        # Main model PR
        y_pred_proba_main = self.main_model.predict_proba(X_test)[:, 1]
        precision_main, recall_main, _ = precision_recall_curve(y_test, y_pred_proba_main)
        
        # Plot curves
        plt.plot(recall_baseline, precision_baseline, label='Baseline (Logistic Regression)', linewidth=2)
        plt.plot(recall_main, precision_main, label='XGBoost', linewidth=2)
        
        # Calibrated model if available
        if self.calibrated_model:
            y_pred_proba_cal = self.calibrated_model.predict_proba(X_test)[:, 1]
            precision_cal, recall_cal, _ = precision_recall_curve(y_test, y_pred_proba_cal)
            plt.plot(recall_cal, precision_cal, label='Calibrated XGBoost', linewidth=2)
        
        # Baseline line (prevalence)
        prevalence = y_test.mean()
        plt.axhline(y=prevalence, color='k', linestyle='--', alpha=0.5, label=f'Baseline (Prevalence = {prevalence:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Diabetes Risk Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pr_path = os.path.join(self.artifacts_dir, 'pr.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-Recall curve saved to: {pr_path}")
        return pr_path
    
    def generate_shap_summary(self, X_train: np.ndarray, sample_size: int = 100) -> str:
        """Generate SHAP summary plot."""
        print("Generating SHAP summary plot...")
        
        try:
            # Sample data for performance (SHAP can be slow)
            if len(X_train) > sample_size:
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_sample = X_train[indices]
            else:
                X_sample = X_train
            
            # Create TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(self.main_model)
            shap_values = explainer.shap_values(X_sample)
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                plot_type='bar',
                show=False
            )
            
            plt.title('SHAP Feature Importance - Diabetes Risk Prediction')
            plt.tight_layout()
            
            shap_path = os.path.join(self.artifacts_dir, 'shap_summary.png')
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP summary plot saved to: {shap_path}")
            return shap_path
            
        except Exception as e:
            print(f"Warning: Could not generate SHAP plot: {e}")
            return None
    
    def run_complete_training(self) -> Dict[str, str]:
        """Run complete training pipeline."""
        print("Starting complete training pipeline...")
        print("="*60)
        
        # Load and split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_split_data()
        
        # Train models
        self.train_baseline_model(X_train, y_train)
        self.train_main_model(X_train, y_train, X_val, y_val)
        
        # Compare models
        is_better = self.compare_models(X_test, y_test)
        
        # Calibrate if needed
        self.calibrate_model(X_train, y_train, X_test, y_test)
        
        saved_files = {}
        
        if is_better:
            print("\n" + "="*50)
            print("SAVING ARTIFACTS (Main model is better)")
            print("="*50)
            
            # Save model and metrics
            saved_files['model'] = self.save_model_pipeline()
            saved_files['metrics'] = self.save_metrics()
            
            # Generate plots
            saved_files['roc'] = self.plot_roc_curve(X_test, y_test)
            saved_files['pr'] = self.plot_precision_recall_curve(X_test, y_test)
            saved_files['shap'] = self.generate_shap_summary(X_train)

            print("\nTraining completed successfully!")
            print("All artifacts saved to ml/artifacts/")
            
        else:
            print("\nMain model did not outperform baseline")
            print("Artifacts not saved")
        
        return saved_files


def main():
    """Main training function."""
    try:
        # Initialize trainer
        trainer = DiabetesTrainer()
        
        # Run training
        saved_files = trainer.run_complete_training()
        
        # Print summary
        if saved_files:
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print("Saved files:")
            for file_type, path in saved_files.items():
                if path:
                    print(f"  {file_type}: {path}")
            
            # Print key metrics
            if trainer.metrics:
                main_metrics = trainer.metrics['main']
                print(f"\nFinal Model Performance:")
                print(f"  ROC AUC: {main_metrics['roc_auc']:.4f}")
                print(f"  F1 Score: {main_metrics['f1']:.4f}")
                print(f"  Precision: {main_metrics['precision']:.4f}")
                print(f"  Recall: {main_metrics['recall']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()