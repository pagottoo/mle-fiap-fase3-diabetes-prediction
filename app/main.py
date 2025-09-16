"""
Dashboard de Avaliação de Indicadores de Saúde para Diabetes
Aplicação Streamlit para predição e análise de risco de diabetes.
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
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional, List

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Avaliação de Indicadores de Diabetes",
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
        return {"success": False, "error": "Cannot connect to API. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API request timeout"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def show_api_status():
    """Display API connection status in sidebar."""
    with st.sidebar:
        st.subheader("🔗 Status da API")
        
        # Test API connection
        result = call_api("/status")
        
        if result["success"]:
            st.success("Conectado")
            st.caption(f"API: {API_URL}")
            
            # Show model status
            data = result["data"]
            st.write("**Status do Modelo:**")
            st.write(f"- ML Disponível: {'Sim' if data.get('ml_available') else 'Não'}")
            st.write(f"- Modelo Treinado: {'Sim' if data.get('model_trained') else 'Não'}")
            
        else:
            st.error("Desconectado")
            st.caption(f"Erro: {result['error']}")


def diabetes_prediction_form():
    """Create the diabetes prediction form."""
    st.markdown('<h1 class="main-header">🩺 Avaliação de Risco de Diabetes</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Predição de Indicadores de Saúde para Diabetes
    Esta ferramenta usa aprendizado de máquina para avaliar o risco de diabetes baseado em vários indicadores de saúde.
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
            help="Selecione sua categoria de idade"
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
                1: "Branca", 2: "Negra", 3: "Asiática", 4: "Indígena Americana/Nativa do Alasca",
                5: "Hispânica", 6: "Outra"
            }[x],
            help="Categoria de raça/etnia"
        )
        
        education = st.selectbox(
            "Nível Educacional",
            options=list(range(1, 7)),
            format_func=lambda x: {
                1: "Nunca frequentou escola", 2: "Ensino Fundamental 1-8",
                3: "Ensino Médio 9-11", 4: "Ensino Médio completo", 
                5: "Ensino Superior 1-3 anos", 6: "Ensino Superior 4+ anos"
            }[x],
            help="Maior nível educacional completado"
        )
        
        income = st.selectbox(
            "Nível de Renda",
            options=list(range(1, 9)),
            format_func=lambda x: {
                1: "Menos de $10,000", 2: "$10,000-$14,999", 3: "$15,000-$19,999",
                4: "$20,000-$24,999", 5: "$25,000-$34,999", 6: "$35,000-$49,999",
                7: "$50,000-$74,999", 8: "$75,000 ou mais"
            }[x],
            help="Faixa de renda anual familiar"
        )
        
        st.subheader("🏥 Medidas de Saúde")
        
        bmi = st.slider(
            "IMC (Índice de Massa Corporal)",
            min_value=10.0, max_value=100.0, value=25.0, step=0.1,
            help="Cálculo do IMC: peso(kg) / altura(m)²"
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
            help="Número de dias em que a saúde física não estava boa"
        )
        
        mental_health = st.slider(
            "Saúde Mental (dias ruins nos últimos 30)",
            min_value=0, max_value=30, value=0, step=1,
            help="Número de dias em que a saúde mental não estava boa"
        )
        
        sleep_time = st.slider(
            "Horas de Sono",
            min_value=1, max_value=24, value=8, step=1,
            help="Média de horas de sono por noite"
        )
        
        st.subheader("🏥 Acesso à Saúde")
        
        any_healthcare = st.selectbox(
            "Tem Plano de Saúde",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Tem algum tipo de cobertura de saúde"
        )
        
        no_docbc_cost = st.selectbox(
            "Deixou de Ver Médico por Custo",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Houve momento em que precisou de médico mas não foi por custo"
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
            help="Bebedores pesados (homens ≥14 drinks/semana, mulheres ≥7 drinks/semana)"
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
            "Teve AVC",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi informado que teve um AVC"
        )
        
        heart_disease_or_attack = st.selectbox(
            "Doença Cardíaca ou Infarto",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Doença coronariana ou infarto do miocárdio"
        )
        
        high_bp = st.selectbox(
            "Pressão Alta",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi informado que tem pressão alta"
        )
        
        high_chol = st.selectbox(
            "Colesterol Alto",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi informado que tem colesterol alto no sangue"
        )
        
        chol_check = st.selectbox(
            "Exame de Colesterol",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Fez exame de colesterol nos últimos 5 anos"
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
            help="Já foi informado que tem asma"
        )
        
        kidney_disease = st.selectbox(
            "Doença Renal",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi informado que tem doença renal"
        )
        
        skin_cancer = st.selectbox(
            "Câncer de Pele",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já foi informado que tem câncer de pele"
        )
        
        diabetic = st.selectbox(
            "Status Pré-diabético",
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
        predict_button = st.button("🔍 Predizer Risco de Diabetes", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare data for prediction
        patient_data = {
            "high_bp": high_bp,
            "high_chol": high_chol,
            "chol_check": chol_check,
            "bmi": bmi,
            "smoker": smoking,
            "stroke": stroke,
            "heart_disease_or_attack": heart_disease_or_attack,
            "phys_activity": physical_activity,
            "fruits": fruits,
            "veggies": veggies,
            "hvy_alcohol_consump": alcohol_drinking,
            "any_healthcare": any_healthcare,
            "no_docbc_cost": no_docbc_cost,
            "gen_hlth": gen_health,
            "ment_hlth": mental_health,
            "phys_hlth": physical_health,
            "diff_walking": diff_walking,
            "sex": sex,
            "age": age_category,
            "education": education,
            "income": income
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
                    <p><strong>Probabilidade de Risco de Diabetes:</strong> {probability:.1%}</p>
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
                            x="abs_shap_value",
                            y="feature",
                            orientation="h",
                            title="Principais Fatores Contribuintes",
                            color="abs_shap_value",
                            color_continuous_scale="RdYlBu_r"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Predição falhou: {result['error']}")


def data_management_page():
    """Data management interface."""
    st.markdown('<h1 class="main-header">📊 Gerenciamento de Dados</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload de Dados", "🎯 Treinamento do Modelo", "📈 Métricas do Modelo", "🔄 Reset do Sistema"])
    
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
                st.success(f"Arquivo enviado com sucesso! Formato: {df.shape}")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Upload to API
                if st.button("📤 Upload to Database"):
                    with st.spinner("Uploading data..."):
                        # Map column names from CSV format to API format
                        column_mapping = {
                            'Diabetes_binary': 'diabetes_binary',
                            'HighBP': 'high_bp',
                            'HighChol': 'high_chol', 
                            'CholCheck': 'chol_check',
                            'BMI': 'bmi',
                            'Smoker': 'smoker',
                            'Stroke': 'stroke',
                            'HeartDiseaseorAttack': 'heart_disease_or_attack',
                            'PhysActivity': 'phys_activity',
                            'Fruits': 'fruits',
                            'Veggies': 'veggies',
                            'HvyAlcoholConsump': 'hvy_alcohol_consump',
                            'AnyHealthcare': 'any_healthcare',
                            'NoDocbcCost': 'no_docbc_cost',
                            'GenHlth': 'gen_hlth',
                            'MentHlth': 'ment_hlth',
                            'PhysHlth': 'phys_hlth',
                            'DiffWalk': 'diff_walking',
                            'Sex': 'sex',
                            'Age': 'age',
                            'Education': 'education',
                            'Income': 'income'
                        }
                        
                        # Apply column mapping
                        df_mapped = df.rename(columns=column_mapping)
                        
                        # Convert DataFrame to records and upload
                        records = df_mapped.to_dict('records')
                        
                        success_count = 0
                        error_count = 0
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, record in enumerate(records):
                            result = call_api("/ingest/record", method="POST", data=record)
                            
                            if result["success"]:
                                success_count += 1
                            else:
                                error_count += 1
                            
                            # Update progress
                            progress = (i + 1) / len(records)
                            progress_bar.progress(progress)
                            status_text.text(f"Processados {i+1}/{len(records)} registros")
                        
                        st.success(f"Upload concluído! Sucessos: {success_count}, Erros: {error_count}")
                        
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    with tab2:
        st.subheader("Treinar Modelo de Predição de Diabetes")
        
        st.markdown("""
        Treine um novo modelo de aprendizado de máquina usando os dados enviados.
        Este processo pode levar vários minutos dependendo do tamanho do dataset.
        """)
        
        if st.button("🎯 Iniciar Treinamento"):
            with st.spinner("Treinando modelo..."):
                result = call_api("/train", method="POST")
            
            if result["success"]:
                training_result = result["data"]
                st.success("Treinamento do modelo concluído!")
                
                # Show data info if available
                if training_result.get("training_records") or training_result.get("data_source"):
                    st.info(f"""
                    **📊 Informações dos Dados de Treinamento:**
                    - **Fonte**: {training_result.get('data_source', 'N/A')}
                    - **Registros para treinamento**: {training_result.get('training_records', 'N/A'):,}
                    - **Tempo de treinamento**: {training_result.get('training_time', 0):.1f}s
                    """)
                
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
                st.error(f"Treinamento falhou: {result['error']}")
    
    with tab3:
        st.subheader("Métricas de Performance do Modelo")
        
        # Get current metrics
        result = call_api("/metrics")
        
        if result["success"] and result["data"].get("metrics"):
            # Use the main model metrics (not baseline)
            metrics_data = result["data"]["metrics"].get("metrics", {})
            main_metrics = metrics_data.get("main", {})
            baseline_metrics = metrics_data.get("baseline", {})
            data_info = result["data"]["metrics"].get("data_info", {})
            
            if main_metrics:
                # Show data information if available
                if data_info:
                    st.success(f"""
                    **📊 Informações do Dataset de Treinamento:**
                    - **Fonte dos dados**: {data_info.get('source', 'N/A')}
                    - **Total de registros**: {data_info.get('total_records', 'N/A'):,}
                    - **Registros de treinamento**: {data_info.get('train_records', 'N/A'):,}
                    - **Registros de validação**: {data_info.get('validation_records', 'N/A'):,}
                    - **Registros de teste**: {data_info.get('test_records', 'N/A'):,}
                    """)
                
                st.info("**Métricas do Modelo Principal (XGBoost)**")
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Acurácia", f"{main_metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precisão", f"{main_metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{main_metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1 Score", f"{main_metrics.get('f1', 0):.3f}")
                
                # Additional metrics row
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("AUC-ROC", f"{main_metrics.get('roc_auc', 0):.3f}")
                with col6:
                    st.metric("Especificidade", f"{main_metrics.get('specificity', 0):.3f}")
                with col7:
                    st.metric("VP", f"{main_metrics.get('true_positives', 0)}")
                with col8:
                    st.metric("VN", f"{main_metrics.get('true_negatives', 0)}")
                
                # Comparison with baseline if available
                if baseline_metrics:
                    st.subheader("📊 Comparação com Baseline")
                    
                    comparison_data = {
                        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
                        'Modelo Principal': [
                            main_metrics.get('accuracy', 0),
                            main_metrics.get('precision', 0), 
                            main_metrics.get('recall', 0),
                            main_metrics.get('f1', 0),
                            main_metrics.get('roc_auc', 0)
                        ],
                        'Baseline': [
                            baseline_metrics.get('accuracy', 0),
                            baseline_metrics.get('precision', 0),
                            baseline_metrics.get('recall', 0), 
                            baseline_metrics.get('f1', 0),
                            baseline_metrics.get('roc_auc', 0)
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
                # Confusion Matrix for main model
                st.subheader("🔍 Matriz de Confusão")
                
                # Build confusion matrix from individual metrics
                tn = main_metrics.get('true_negatives', 0)
                fp = main_metrics.get('false_positives', 0) 
                fn = main_metrics.get('false_negatives', 0)
                tp = main_metrics.get('true_positives', 0)
                
                cm = np.array([[tn, fp], [fn, tp]])
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Confusão - Modelo Principal",
                    labels=dict(x="Predito", y="Real"),
                    x=['Sem Diabetes', 'Com Diabetes'],
                    y=['Sem Diabetes', 'Com Diabetes'],
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Verdadeiros Negativos (VN)", tn)
                    st.metric("Falsos Negativos (FN)", fn)
                with col2:
                    st.metric("Falsos Positivos (FP)", fp)
                    st.metric("Verdadeiros Positivos (VP)", tp)
        else:
            st.info("Métricas do modelo não disponíveis. Treine um modelo primeiro.")
    
    with tab4:
        st.subheader("🔄 Reset Completo do Sistema")
        
        st.markdown("""
        ⚠️ **ATENÇÃO**: Esta operação irá **apagar todos os dados** e **remover o modelo treinado**.
        
        Use esta funcionalidade para:
        - 🧹 Limpar completamente o sistema
        - 🔄 Começar do zero com novos dados  
        - 🧪 Experimentar com datasets diferentes
        - 🐛 Resolver problemas de corrupção de dados
        """)
        
        # Reset options
        st.markdown("### Opções de Reset")
        
        col1, col2 = st.columns(2)
        with col1:
            reset_data = st.checkbox(
                "🗃️ Limpar dados do banco",
                value=True,
                help="Remove todos os registros das tabelas diabetes_raw e models"
            )
        with col2:
            reset_model = st.checkbox(
                "🤖 Remover modelo treinado",
                value=True, 
                help="Remove arquivos model.pkl, metrics.json e outros artefatos"
            )
        
        if not (reset_data or reset_model):
            st.warning("⚠️ Selecione pelo menos uma opção de reset.")
        else:
            # Safety confirmation
            st.markdown("### 🛡️ Confirmação de Segurança")
            
            confirm_text = st.text_input(
                "Digite 'RESET' para confirmar:",
                placeholder="RESET",
                help="Digite exatamente 'RESET' em maiúsculas para confirmar"
            )
            
            if confirm_text == "RESET":
                st.success("✅ Confirmação válida")
                
                # Reset button
                if st.button("🔄 **EXECUTAR RESET COMPLETO**", type="primary"):
                    with st.spinner("Executando reset do sistema..."):
                        # Call reset API
                        reset_data_payload = {
                            "confirm": True,
                            "reset_data": reset_data,
                            "reset_model": reset_model
                        }
                        
                        result = call_api("/reset", method="POST", data=reset_data_payload)
                    
                    if result["success"]:
                        reset_result = result["data"]
                        
                        st.success("🎉 **Reset executado com sucesso!**")
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Registros Removidos", reset_result.get("deleted_records", 0))
                        with col2:
                            st.metric("🤖 Modelos Removidos", reset_result.get("deleted_models", 0))
                        with col3:
                            artifacts = reset_result.get("artifacts_removed", [])
                            st.metric("📁 Artefatos Removidos", len(artifacts))
                        
                        if artifacts:
                            st.info(f"**Artefatos removidos:** {', '.join(artifacts)}")
                        
                        # Instructions for next steps
                        st.markdown("""
                        ### 🚀 Próximos Passos:
                        1. **Upload novos dados** na aba "📤 Upload de Dados"
                        2. **Treinar novo modelo** na aba "🎯 Treinamento do Modelo" 
                        3. **Verificar métricas** na aba "📈 Métricas do Modelo"
                        """)
                        
                        # Auto-refresh after reset to update UI
                        st.rerun()
                    else:
                        st.error(f"❌ Reset falhou: {result['error']}")
            
            elif confirm_text and confirm_text != "RESET":
                st.error("❌ Confirmação inválida. Digite exatamente 'RESET'")
            else:
                st.info("🔒 Digite 'RESET' para habilitar o botão de reset")


def data_analysis_page():
    """Data analysis page with matplotlib visualizations."""
    st.markdown('<h1 class="main-header">📈 Análise de Dados de Diabetes</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Visualizações Interativas dos Dados de Saúde
    Esta página oferece análises visuais dos dados (amostragem) de diabetes usando gráficos estatísticos.
    """)
    
    # Get data from API
    with st.spinner("Carregando dados..."):
        result = call_api("/records", params={"limit": 10000})
        
        if not result["success"]:
            st.error(f"Erro ao carregar dados: {result['error']}")
            st.info("Certifique-se de que há dados no banco e que a API está funcionando.")
            return
        
        data = result["data"]
        records = data.get("records", [])
        if not records or len(records) == 0:
            st.warning("Nenhum dado encontrado no banco de dados.")
            st.info("Importe dados na página 'Gerenciamento de Dados' primeiro.")
            return
        
        df = pd.DataFrame(records)
        st.success(f"Ok: {len(df)} registros carregados com sucesso!")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filtros de Análise")
        
        # Age filter
        age_options = sorted(df['age'].unique())
        age_range = st.slider(
            "Faixa Etária",
            min_value=int(min(age_options)),
            max_value=int(max(age_options)),
            value=(int(min(age_options)), int(max(age_options))),
            help="Selecione a faixa etária para análise"
        )
        
        # BMI filter
        bmi_range = st.slider(
            "Índice de Massa Corporal (BMI)",
            min_value=float(df['bmi'].min()),
            max_value=float(df['bmi'].max()),
            value=(float(df['bmi'].min()), float(df['bmi'].max())),
            format="%.1f",
            help="Selecione a faixa de BMI para análise"
        )
        
        # Sex filter
        sex_filter = st.selectbox(
            "Sexo",
            options=["Todos", "Feminino", "Masculino"],
            help="Filtrar por sexo"
        )
    
    # Apply filters
    filtered_df = df[
        (df['age'] >= age_range[0]) & 
        (df['age'] <= age_range[1]) &
        (df['bmi'] >= bmi_range[0]) & 
        (df['bmi'] <= bmi_range[1])
    ]
    
    if sex_filter != "Todos":
        sex_value = 0 if sex_filter == "Feminino" else 1
        filtered_df = filtered_df[filtered_df['sex'] == sex_value]
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados.")
        return
    
    # Display filtered data metrics
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric("Total de Registros", len(filtered_df))
    
    with col_info2:
        diabetes_rate = (filtered_df['diabetes_binary'].mean() * 100)
        st.metric("Taxa de Diabetes", f"{diabetes_rate:.1f}%")
    
    with col_info3:
        avg_bmi = filtered_df['bmi'].mean()
        st.metric("BMI Médio", f"{avg_bmi:.1f}")
    
    with col_info4:
        high_risk = filtered_df[
            (filtered_df['high_bp'] == 1) & 
            (filtered_df['high_chol'] == 1)
        ]['diabetes_binary'].mean() * 100
        st.metric("Alto Risco (%)", f"{high_risk:.1f}%")
    
    # Create visualizations
    st.markdown("---")
    
    # Row 1: BMI Distribution and Risk Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de BMI por Status de Diabetes")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Create histograms
        no_diabetes = filtered_df[filtered_df['diabetes_binary'] == 0]['bmi']
        with_diabetes = filtered_df[filtered_df['diabetes_binary'] == 1]['bmi']
        
        # Only create histogram if we have data
        if len(no_diabetes) > 0:
            ax1.hist(no_diabetes, alpha=0.7, label='Sem Diabetes', bins=min(30, len(no_diabetes)//2 + 1), color='lightblue', density=True)
        if len(with_diabetes) > 0:
            ax1.hist(with_diabetes, alpha=0.7, label='Com Diabetes', bins=min(30, len(with_diabetes)//2 + 1), color='salmon', density=True)
        
        ax1.set_xlabel('Índice de Massa Corporal (BMI)')
        ax1.set_ylabel('Densidade')
        ax1.set_title('Distribuição de BMI por Status de Diabetes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("🎯 Taxa de Diabetes por Fatores de Risco")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Risk factors analysis
        risk_factors = {
            'Pressão Alta': 'high_bp',
            'Colesterol Alto': 'high_chol', 
            'Fumante': 'smoker',
            'AVC': 'stroke',
            'Doença Cardíaca': 'heart_disease_or_attack',
            'Atividade Física': 'phys_activity'
        }
        
        diabetes_rates = []
        factor_names = []
        
        for name, column in risk_factors.items():
            if column in filtered_df.columns:
                rate = filtered_df[filtered_df[column] == 1]['diabetes_binary'].mean() * 100
                diabetes_rates.append(rate)
                factor_names.append(name)
        
        bars = ax2.bar(factor_names, diabetes_rates, color='coral')
        ax2.set_title('Taxa de Diabetes por Fator de Risco (%)')
        ax2.set_ylabel('Taxa de Diabetes (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, rate in zip(bars, diabetes_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Row 2: Age Distribution and Correlation Heatmap
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👥 Distribuição por Faixa Etária")
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # Age distribution by diabetes status
        age_diabetes = filtered_df.groupby(['age', 'diabetes_binary']).size().unstack(fill_value=0)
        
        age_diabetes.plot(kind='bar', ax=ax3, color=['lightblue', 'salmon'], width=0.8)
        ax3.set_title('Distribuição de Casos por Faixa Etária')
        ax3.set_xlabel('Faixa Etária')
        ax3.set_ylabel('Número de Casos')
        ax3.legend(['Sem Diabetes', 'Com Diabetes'])
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    
    with col4:
        st.subheader("🔥 Correlação entre Indicadores")
        
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        # Remove ID columns if present
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        if len(numeric_cols) > 1:
            correlation_matrix = filtered_df[numeric_cols].corr()
            
            # Create heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax4, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax4.set_title('Matriz de Correlação - Indicadores de Saúde')
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("Dados insuficientes para matriz de correlação.")
    
    # Row 3: Advanced Analysis
    st.markdown("---")
    st.subheader("📈 Análise Avançada")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("🎻 Distribuição de BMI (Violin Plot)")
        
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        
        # Prepare data for violin plot
        diabetes_labels = ['Sem Diabetes', 'Com Diabetes']
        bmi_no_diabetes = filtered_df[filtered_df['diabetes_binary'] == 0]['bmi'].values
        bmi_with_diabetes = filtered_df[filtered_df['diabetes_binary'] == 1]['bmi'].values
        
        # Check if we have enough data for violin plot
        if len(bmi_no_diabetes) > 5 and len(bmi_with_diabetes) > 5:
            bmi_data = [bmi_no_diabetes, bmi_with_diabetes]
            
            # Create violin plot
            parts = ax5.violinplot(bmi_data, positions=[0, 1], showmeans=True, showmedians=True)
            
            # Customize colors
            colors = ['lightblue', 'salmon']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            ax5.set_xticks([0, 1])
            ax5.set_xticklabels(diabetes_labels)
            ax5.set_ylabel('BMI')
            ax5.set_title('Distribuição de BMI por Status de Diabetes')
            ax5.grid(True, alpha=0.3)
        else:
            # Fallback to histogram if not enough data for violin plot
            if len(bmi_no_diabetes) > 0:
                ax5.hist(bmi_no_diabetes, alpha=0.7, label='Sem Diabetes', 
                        bins=min(20, max(5, len(bmi_no_diabetes)//3)), color='lightblue')
            if len(bmi_with_diabetes) > 0:
                ax5.hist(bmi_with_diabetes, alpha=0.7, label='Com Diabetes', 
                        bins=min(20, max(5, len(bmi_with_diabetes)//3)), color='salmon')
            
            ax5.set_xlabel('BMI')
            ax5.set_ylabel('Frequência')
            ax5.set_title('Distribuição de BMI por Status de Diabetes')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        st.pyplot(fig5)
        plt.close(fig5)
    
    with col6:
        st.subheader("📋 Resumo Estatístico")
        
        # Statistical summary
        summary_stats = filtered_df.groupby('diabetes_binary')['bmi'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        # Map index to labels (handle cases where one group might be missing)
        index_mapping = {0: 'Sem Diabetes', 1: 'Com Diabetes'}
        summary_stats.index = [index_mapping[i] for i in summary_stats.index]
        summary_stats.columns = ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Additional insights
        st.markdown("**💡 Insights:**")
        
        # Only calculate BMI difference if both groups exist
        if 'Com Diabetes' in summary_stats.index and 'Sem Diabetes' in summary_stats.index:
            bmi_diff = summary_stats.loc['Com Diabetes', 'Média'] - summary_stats.loc['Sem Diabetes', 'Média']
            
            if bmi_diff > 0:
                st.write(f"• Pessoas com diabetes têm BMI médio {bmi_diff:.1f} pontos maior")
            else:
                st.write(f"• Pessoas sem diabetes têm BMI médio {abs(bmi_diff):.1f} pontos maior")
        else:
            st.write("• Dados insuficientes para comparação de BMI entre grupos")
        
        diabetes_percentage = (filtered_df['diabetes_binary'].sum() / len(filtered_df)) * 100
        st.write(f"• {diabetes_percentage:.1f}% dos registros filtrados têm diabetes")
        
        if 'phys_activity' in filtered_df.columns:
            active_diabetes_rate = filtered_df[filtered_df['phys_activity'] == 1]['diabetes_binary'].mean() * 100
            inactive_diabetes_rate = filtered_df[filtered_df['phys_activity'] == 0]['diabetes_binary'].mean() * 100
            
            if active_diabetes_rate < inactive_diabetes_rate:
                st.write(f"• Pessoas ativas têm {inactive_diabetes_rate - active_diabetes_rate:.1f}% menos diabetes")
    
    # Footer with data info
    st.markdown("---")
    st.caption(f"📊 Análise baseada em {len(filtered_df)} registros filtrados de {len(df)} registros totais")


def main():
    """Main application function."""
    # Show API status in sidebar
    show_api_status()
    
    # Navigation
    with st.sidebar:
        st.markdown("---")
        page = st.selectbox(
            "Navegação",
            ["🩺 Predição de Diabetes", "📊 Gerenciamento de Dados", "📈 Análise de Dados"],
            key="page_selector"
        )
    
    # Route to appropriate page
    if page == "🩺 Predição de Diabetes":
        diabetes_prediction_form()
    elif page == "📊 Gerenciamento de Dados":
        data_management_page()
    elif page == "📈 Análise de Dados":
        data_analysis_page()


if __name__ == "__main__":
    main()