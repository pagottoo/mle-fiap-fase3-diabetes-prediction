#!/usr/bin/env python3
"""
Script para download automático dos dados de diabetes.
Facilita a configuração inicial do projeto.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_data():
    """
    Download dos dados de diabetes do Kaggle.
    
    Nota: Este script serve como exemplo. Para uso real, você precisará:
    1. Instalar kaggle CLI: pip install kaggle
    2. Configurar API token do Kaggle
    3. Executar: kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset
    """
    
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Script de Download dos Dados de Diabetes")
    print("=" * 50)
    
    # Verificar se já existem dados
    main_file = data_dir / "diabetes.csv"
    if main_file.exists():
        print("Dados já existem em data/diabetes.csv")
        print("Sistema pronto para usar!")
        return
    
    print("Dados não encontrados na pasta data/")
    print("\nPara baixar os dados, siga estas opções:\n")
    
    print("OPÇÃO 1 - Via Kaggle CLI (Recomendado):")
    print("   1. pip install kaggle")
    print("   2. Configure sua API key do Kaggle")
    print("   3. kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset")
    print("   4. Extraia o zip na pasta data/")
    
    print("\nOPÇÃO 2 - Download Manual:")
    print("   1. Acesse: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
    print("   2. Clique em 'Download' (precisa estar logado)")
    print("   3. Extraia todos os arquivos CSV na pasta data/")
    
    print("\nARQUIVOS NECESSÁRIOS:")
    required_files = [
        "diabetes.csv",
        "diabetes_binary_health_indicators_BRFSS2015.csv", 
        "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "diabetes_012_health_indicators_BRFSS2015.csv"
    ]
    
    for file in required_files:
        status = "Ok" if (data_dir / file).exists() else "Error"
        print(f"   {status} {file}")

    print(f"\nPasta de destino: {data_dir.absolute()}")
    print("\nApós baixar os dados, execute: docker compose up --build -d")

if __name__ == "__main__":
    download_data()
