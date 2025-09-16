"""
Diabetes Health Indicators Assessment Dashboard
Streamlit application for diabetes risk prediction and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional, List

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Diabetes Health Indicators Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def call_api(endpoint: str, method: str = "GET", data: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Make API call to backend service."""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, params=params, timeout=30)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"API Error {response.status_code}: {response.text}"
            }
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Não foi possível conectar à API. O servidor está rodando?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout da requisição à API"}
    except Exception as e:
        return {"success": False, "error": f"Erro inesperado: {str(e)}"}
def show_api_status():
    """Display API connection status in sidebar."""
    with st.sidebar:
        st.subheader("🔗 Status da API")
        
        # Test API connection
        result = call_api("/status")
        
        if result["success"]:
            st.success("OK - Conectado")
            st.caption(f"API: {API_URL}")
            
            # Show model status
            data = result["data"]
            st.write("**Status do Modelo:**")
            st.write(f"- ML Disponível: {'Sim' if data.get('ml_available') else 'Não'}")
            st.write(f"- Modelo Treinado: {'Sim' if data.get('model_trained') else 'Não'}")
            
        else:
            st.error("Erro! Desconectado")
            st.caption(f"Erro: {result['error']}")


def diabetes_prediction_form():
    """Create the diabetes prediction form."""
    st.markdown('<h1 class="main-header">🩺 Avaliação de Risco de Diabetes</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Predição de Indicadores de Saúde para Diabetes
    Esta ferramenta usa machine learning para avaliar o risco de diabetes baseado em vários indicadores de saúde.
    Preencha o formulário abaixo para obter uma avaliação personalizada de risco.
    """)
    
    # Create form columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informações Demográficas")
        
        age_category = st.selectbox(
            "Categoria de Idade",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
                6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
                11: "70-74", 12: "75-79", 13: "80+"
            }[x],
            help="Selecione sua faixa etária"
        )
        
        sex = st.selectbox(
            "Sexo",
            options=[0, 1],
            format_func=lambda x: "Feminino" if x == 0 else "Masculino",
            help="Sexo biológico"
        )
        
        race = st.selectbox(
            "Raça/Etnia",
            options=list(range(1, 7)),
            format_func=lambda x: {
                1: "Branco", 2: "Negro", 3: "Asiático", 4: "Indígena Americano/Nativo do Alasca",
                5: "Hispânico", 6: "Outro"
            }[x],
            help="Categoria de raça/etnia"
        )
        
        st.subheader("🏥 Medições de Saúde")
        
        bmi = st.slider(
            "IMC (Índice de Massa Corporal)",
            min_value=10.0, max_value=100.0, value=25.0, step=0.1,
            help="Cálculo do Índice de Massa Corporal: peso(kg) / altura(m)²"
        )
        
        gen_health = st.selectbox(
            "Saúde Geral",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Excelente", 2: "Muito Boa", 3: "Boa", 4: "Regular", 5: "Ruim"}[x],
            help="Avaliação geral da saúde"
        )
        
        physical_health = st.slider(
            "Saúde Física (dias ruins nos últimos 30)",
            min_value=0, max_value=30, value=0, step=1,
            help="Número de dias em que a saúde física não esteve boa"
        )
        
        mental_health = st.slider(
            "Saúde Mental (dias ruins nos últimos 30)",
            min_value=0, max_value=30, value=0, step=1,
            help="Número de dias em que a saúde mental não esteve boa"
        )
        
        sleep_time = st.slider(
            "Horas de Sono",
            min_value=1, max_value=24, value=8, step=1,
            help="Média de horas de sono por noite"
        )
    
    with col2:
        st.subheader("🚭 Fatores de Estilo de Vida")
        
        smoking = st.selectbox(
            "Tabagismo",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Você fumou pelo menos 100 cigarros em toda sua vida?"
        )
        
        alcohol_drinking = st.selectbox(
            "Consumo Excessivo de Álcool",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Consumo excessivo (homens ≥14 drinks/semana, mulheres ≥7 drinks/semana)"
        )
        
        physical_activity = st.selectbox(
            "Atividade Física",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Atividade física nos últimos 30 dias (não incluindo trabalho)"
        )
        
        fruits = st.selectbox(
            "Consumo de Frutas",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Consome frutas 1 ou mais vezes por dia"
        )
        
        veggies = st.selectbox(
            "Consumo de Vegetais",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",  
            help="Consome vegetais 1 ou mais vezes por dia"
        )
        
        st.subheader("🏥 Condições Médicas")
        
        stroke = st.selectbox(
            "AVC (Derrame)",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com AVC (derrame cerebral)?"
        )
        
        heart_disease_or_attack = st.selectbox(
            "Doença Cardíaca ou Ataque",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Doença coronariana ou infarto do miocárdio"
        )
        
        high_bp = st.selectbox(
            "Pressão Alta",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com pressão arterial alta?"
        )
        
        high_chol = st.selectbox(
            "Colesterol Alto",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com colesterol alto?"
        )
        
        chol_check = st.selectbox(
            "Exame de Colesterol",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Fez exame de colesterol nos últimos 5 anos?"
        )
        
        diff_walking = st.selectbox(
            "Dificuldade para Caminhar",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Dificuldade séria para caminhar ou subir escadas"
        )
        
        asthma = st.selectbox(
            "Asma",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com asma?"
        )
        
        kidney_disease = st.selectbox(
            "Doença Renal",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com doença nos rins?"
        )
        
        skin_cancer = st.selectbox(
            "Câncer de Pele",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi diagnosticado com câncer de pele?"
        )
        
        diabetic = st.selectbox(
            "Status Diabético",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Não", 1: "Pré-diabetes", 2: "Sim", 3: "Sim, durante gravidez", 4: "Não, diabetes limítrofe"
            }[x],
            help="Condição diabética pré-existente"
        )
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        predict_button = st.button("🔍 Avaliar Risco de Diabetes", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare data for prediction
        patient_data = {
            "bmi": bmi,
            "smoking": smoking,
            "alcohol_drinking": alcohol_drinking,
            "stroke": stroke,
            "physical_health": physical_health,
            "mental_health": mental_health,
            "diff_walking": diff_walking,
            "sex": sex,
            "age_category": age_category,
            "race": race,
            "diabetic": diabetic,
            "physical_activity": physical_activity,
            "gen_health": gen_health,
            "sleep_time": sleep_time,
            "asthma": asthma,
            "kidney_disease": kidney_disease,
            "skin_cancer": skin_cancer,
            "heart_disease_or_attack": heart_disease_or_attack,
            "high_bp": high_bp,
            "high_chol": high_chol,
            "chol_check": chol_check,
            "fruits": fruits,
            "veggies": veggies
        }
        
        # Make prediction
        with st.spinner("Analisando indicadores de saúde..."):
            result = call_api("/predict", method="POST", data=patient_data)
        
        if result["success"]:
            prediction = result["data"]
            
            # Display results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                probability = prediction["probability"]
                risk_level = prediction["risk_level"]
                class_prediction = prediction["class_prediction"]
                
                # Create risk level styling
                if risk_level.lower() == "high":
                    risk_class = "risk-high"
                    risk_emoji = "🔴"
                elif risk_level.lower() == "medium":
                    risk_class = "risk-medium"
                    risk_emoji = "🟡"
                else:
                    risk_class = "risk-low"
                    risk_emoji = "🟢"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3>{risk_emoji} Nível de Risco: {risk_level}</h3>
                    <p><strong>Probabilidade de Diabetes:</strong> {probability:.1%}</p>
                    <p><strong>Predição:</strong> {'Risco de Diabetes' if class_prediction == 1 else 'Sem Risco de Diabetes'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top features if available
                if prediction.get("top_features"):
                    st.subheader("📊 Principais Fatores de Risco")
                    
                    features_df = pd.DataFrame(prediction["top_features"])
                    if not features_df.empty:
                        fig = px.bar(
                            features_df,
                            x="importance",
                            y="feature",
                            orientation="h",
                            title="Principais Fatores Contribuintes",
                            color="importance",
                            color_continuous_scale="RdYlBu_r"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Falha na predição: {result['error']}")


def data_management_page():
    """Data management interface."""
    st.markdown('<h1 class="main-header">📊 Gerenciamento de Dados</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload de Dados", "🎯 Treinamento do Modelo", "📈 Métricas do Modelo"])
    
    with tab1:
        st.subheader("Upload de Dados de Diabetes")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type="csv",
            help="Faça upload de um arquivo CSV com dados de indicadores de saúde para diabetes"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Arquivo carregado com sucesso! Dimensões: {df.shape}")
                
                # Show preview
                st.subheader("Prévia dos Dados")
                st.dataframe(df.head())
                
                # Upload to API
                if st.button("📤 Enviar para Banco de Dados"):
                    with st.spinner("Enviando dados..."):
                        # Convert DataFrame to records and upload
                        records = df.to_dict('records')
                        
                        success_count = 0
                        error_count = 0
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, record in enumerate(records):
                            result = call_api("/ingest", method="POST", data=record)
                            
                            if result["success"]:
                                success_count += 1
                            else:
                                error_count += 1
                            
                            # Update progress
                            progress = (i + 1) / len(records)
                            progress_bar.progress(progress)
                            status_text.text(f"Processados {i+1}/{len(records)} registros")
                        
                        st.success(f"Upload concluído! Sucesso: {success_count}, Erros: {error_count}")
                        
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    with tab2:
        st.subheader("Treinar Modelo de Predição de Diabetes")
        
        st.markdown("""
        Treine um novo modelo de machine learning usando os dados carregados.
        Este processo pode levar vários minutos dependendo do tamanho do dataset.
        """)
        
        if st.button("🎯 Iniciar Treinamento"):
            with st.spinner("Treinando modelo..."):
                result = call_api("/train", method="POST")
            
            if result["success"]:
                training_result = result["data"]
                st.success("Treinamento do modelo concluído!")
                
                if training_result.get("metrics"):
                    st.subheader("Resultados do Treinamento")
                    metrics = training_result["metrics"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Acurácia", f"{metrics.get('accuracy', 0):.3f}")
                    with col2:
                        st.metric("Precisão", f"{metrics.get('precision', 0):.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
            else:
                st.error(f"Falha no treinamento: {result['error']}")
    
    with tab3:
        st.subheader("Métricas de Performance do Modelo")
        
        # Get current metrics
        result = call_api("/metrics")
        
        if result["success"] and result["data"]["metrics"]:
            metrics = result["data"]["metrics"]
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Acurácia", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precisão", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
            
            # Confusion matrix
            if 'confusion_matrix' in metrics:
                st.subheader("Matriz de Confusão")
                cm = np.array(metrics['confusion_matrix'])
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Confusão",
                    labels=dict(x="Predito", y="Real"),
                    x=['Sem Diabetes', 'Com Diabetes'],
                    y=['Sem Diabetes', 'Com Diabetes']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Métricas do modelo não disponíveis. Treine um modelo primeiro.")


def main():
    """Main application function."""
    # Show API status in sidebar
    show_api_status()
    
    # Navigation
    with st.sidebar:
        st.markdown("---")
        page = st.selectbox(
            "Navegação",
            ["🩺 Predição de Diabetes", "📊 Gerenciamento de Dados"],
            key="page_selector"
        )
    
    # Route to appropriate page
    if page == "🩺 Predição de Diabetes":
        diabetes_prediction_form()
    elif page == "📊 Gerenciamento de Dados":
        data_management_page()


if __name__ == "__main__":
    main()