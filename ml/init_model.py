#!/usr/bin/env python3
"""
Simple model initialization script for Docker deployment.
Ensures the model is trained and ready before the API starts.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.train import DiabetesTrainer

def init_model():
    """Initialize model if not already trained."""
    artifacts_dir = Path("./ml/artifacts")
    model_path = artifacts_dir / "model.pkl"
    
    if not model_path.exists():
        print("🚀 Model not found. Training new model...")
        try:
            # Initialize trainer with default paths
            trainer = DiabetesTrainer()
            
            # Train the model
            print("Loading and preprocessing data...")
            trainer.load_data()
            trainer.preprocess()

            print("Training XGBoost model...")
            trainer.train_xgboost()

            print("Saving model and metrics...")
            trainer.save_model()
            trainer.save_metrics()

            print("Model training completed successfully!")

        except Exception as e:
            print(f"Model training failed: {e}")
            sys.exit(1)
    else:
        print("Model already exists. Skipping training.")

if __name__ == "__main__":
    init_model()