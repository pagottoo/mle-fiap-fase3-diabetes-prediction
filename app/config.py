"""
Configuration file for the Streamlit Diabetes Health Indicators App.
"""

import os
from typing import Dict, Any

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Diabetes Health Indicators Assessment",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Feature descriptions and configurations
FEATURE_CONFIG = {
    "age": {
        "display_name": "Idade",
        "type": "slider",
        "min_value": 1,
        "max_value": 120,
        "default": 50,
        "step": 1,
        "unit": "anos",
        "help": "Idade do paciente em anos"
    },
    "sex": {
        "display_name": "Sexo",
        "type": "selectbox", 
        "options": [0, 1],
        "labels": {0: "Feminino", 1: "Masculino"},
        "help": "Sexo biológico do paciente"
    },
    "cp": {
        "display_name": "Tipo de Dor no Peito",
        "type": "selectbox",
        "options": [0, 1, 2, 3],
        "labels": {
            0: "Assintomática",
            1: "Angina Atípica",
            2: "Dor Não-Anginosa", 
            3: "Angina Típica"
        },
        "help": "Tipo de dor torácica relatada"
    },
    "trestbps": {
        "display_name": "Pressão Arterial em Repouso",
        "type": "slider",
        "min_value": 80,
        "max_value": 250,
        "default": 120,
        "step": 5,
        "unit": "mmHg",
        "help": "Pressão arterial sistólica em repouso"
    },
    "chol": {
        "display_name": "Colesterol Sérico",
        "type": "slider",
        "min_value": 100,
        "max_value": 600,
        "default": 200,
        "step": 10,
        "unit": "mg/dl",
        "help": "Nível de colesterol no sangue"
    },
    "fbs": {
        "display_name": "Açúcar no Sangue em Jejum > 120 mg/dl",
        "type": "selectbox",
        "options": [0, 1],
        "labels": {0: "Não", 1: "Sim"},
        "help": "Glicemia de jejum elevada"
    },
    "restecg": {
        "display_name": "Resultados ECG em Repouso",
        "type": "selectbox",
        "options": [0, 1, 2],
        "labels": {
            0: "Normal",
            1: "Anormalidade ST-T",
            2: "Hipertrofia Ventricular"
        },
        "help": "Resultados do eletrocardiograma em repouso"
    },
    "thalach": {
        "display_name": "Frequência Cardíaca Máxima",
        "type": "slider",
        "min_value": 60,
        "max_value": 220,
        "default": 150,
        "step": 5,
        "unit": "bpm",
        "help": "Frequência cardíaca máxima atingida"
    },
    "exang": {
        "display_name": "Angina Induzida por Exercício",
        "type": "selectbox",
        "options": [0, 1],
        "labels": {0: "Não", 1: "Sim"},
        "help": "Angina causada por exercício físico"
    },
    "oldpeak": {
        "display_name": "Depressão ST",
        "type": "slider",
        "min_value": 0.0,
        "max_value": 10.0,
        "default": 1.0,
        "step": 0.1,
        "unit": "",
        "help": "Depressão do segmento ST induzida por exercício"
    },
    "slope": {
        "display_name": "Inclinação do Segmento ST",
        "type": "selectbox",
        "options": [0, 1, 2],
        "labels": {
            0: "Descendente",
            1: "Plana",
            2: "Ascendente"
        },
        "help": "Inclinação do segmento ST no pico do exercício"
    },
    "ca": {
        "display_name": "Vasos Principais (fluoroscopia)",
        "type": "selectbox",
        "options": [0, 1, 2, 3, 4],
        "labels": {0: "0 vasos", 1: "1 vaso", 2: "2 vasos", 3: "3 vasos", 4: "4 vasos"},
        "help": "Número de vasos principais coloridos por fluoroscopia"
    },
    "thal": {
        "display_name": "Talassemia",
        "type": "selectbox",
        "options": [0, 1, 2, 3],
        "labels": {
            0: "Não informado",
            1: "Defeito Fixo",
            2: "Normal",
            3: "Defeito Reversível"
        },
        "help": "Tipo de talassemia detectada"
    }
}

# Risk level colors and styling
RISK_LEVEL_CONFIG = {
    "Low": {
        "color": "#4CAF50",
        "background": "#E8F5E8",
        "border": "#4CAF50",
        "icon": "✅",
        "message": "Paciente apresenta baixo risco de doença cardíaca."
    },
    "Medium": {
        "color": "#FF9800", 
        "background": "#FFF3E0",
        "border": "#FF9800",
        "icon": "ℹ️",
        "message": "Paciente apresenta risco moderado. Monitoramento e acompanhamento médico recomendados."
    },
    "High": {
        "color": "#F44336",
        "background": "#FFEBEE", 
        "border": "#F44336",
        "icon": "⚠️",
        "message": "Paciente apresenta alta probabilidade de doença cardíaca. Recomenda-se avaliação médica urgente."
    }
}

# Chart configurations
CHART_CONFIG = {
    "gauge": {
        "height": 300,
        "color_scale": {
            "low": "#E8F5E8",
            "medium": "#FFF3E0", 
            "high": "#FFEBEE"
        }
    },
    "roc_curve": {
        "height": 400,
        "title": "ROC Curve"
    },
    "pr_curve": {
        "height": 400,
        "title": "Precision-Recall Curve" 
    },
    "confusion_matrix": {
        "height": 400,
        "color_scale": "Blues"
    }
}

# File paths
ARTIFACT_PATHS = {
    "model": "ml/artifacts/model.pkl",
    "metrics": "ml/artifacts/metrics.json",
    "roc_plot": "ml/artifacts/roc.png",
    "pr_plot": "ml/artifacts/pr.png", 
    "shap_summary": "ml/artifacts/shap_summary.png"
}

# API endpoints
API_ENDPOINTS = {
    "health": "/health",
    "predict": "/predict",
    "train": "/train",
    "metrics": "/metrics",
    "records": "/records",
    "ingest": "/ingest/record",
    "status": "/status"
}

# Default values for forms
DEFAULT_PATIENT = {
    "age": 50,
    "sex": 1,
    "cp": 1,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# UI Messages
MESSAGES = {
    "api_connecting": "Conectando à API...",
    "api_connected": "Done - Conectado à API",
    "api_disconnected": "Desconectado da API",
    "loading_data": "Carregando dados...",
    "prediction_running": "Analisando dados do paciente...",
    "training_running": "Treinando modelo...",
    "no_model": "Modelo não encontrado. Execute o treinamento primeiro.",
    "no_data": "Dados não disponíveis.",
    "error_generic": "Ocorreu um erro inesperado."
}

def get_feature_config(feature_name: str) -> Dict[str, Any]:
    """Get configuration for a specific feature."""
    return FEATURE_CONFIG.get(feature_name, {})

def get_risk_config(risk_level: str) -> Dict[str, str]:
    """Get styling configuration for a risk level."""
    return RISK_LEVEL_CONFIG.get(risk_level, RISK_LEVEL_CONFIG["Medium"])

def get_api_url(endpoint: str) -> str:
    """Get full API URL for an endpoint."""
    return f"{API_BASE_URL}{API_ENDPOINTS.get(endpoint, endpoint)}"