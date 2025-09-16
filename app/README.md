# Sistema de Predição de Diabetes - Interface Web com Streamlit

## Visão Geral

Interface web completa desenvolvida em Streamlit contendo:
- **Predições de diabetes** com explicabilidade SHAP
- **Análises visuais** com matplotlib e seaborn
- **Gerenciamento de dados** com upload CSV
- **Filtros interativos** para exploração de dados

## Links Uteis:

- **URL:** http://localhost:8501
- **Páginas Disponíveis:**
  - **Predição de Diabetes** - Avaliação de risco individual
  - **Gerenciamento de Dados** - Upload e treinamento
  - **Análise de Dados** - Visualizações dos dados

## Páginas e Funcionalidades

### **1. Predição de Diabetes**

**Formulário com os 21 campos (features) de saúde:**

#### **Informações Demográficas**
- **Categoria de Idade:** 1-13 (18-24 até 80+)
- **Sexo:** Feminino/Masculino
- **Raça:** 8 categorias
- **Educação:** 6 níveis
- **Renda:** 8 faixas de renda

#### **Indicadores de Saúde**
- **BMI:** Índice de Massa Corporal (10-100)
- **Saúde Geral:** Escala 1-5 (Excelente a Ruim)
- **Pressão Arterial Alta:** Sim/Não
- **Colesterol Alto:** Sim/Não
- **Checkup de Colesterol:** Sim/Não

#### **Estilo de Vida**
- **Fumante:** Sim/Não
- **Atividade Física:** Últimos 30 dias
- **Consumo de Frutas:** Diário
- **Consumo de Vegetais:** Diário
- **Consumo Pesado de Álcool:** Sim/Não

### **2. Gerenciamento de Dados**

#### **Upload de Dados**
- **Formato:** CSV com colunas mapeadas automaticamente
- **Validação:** Verificação de tipos e valores
- **Feedback:** Contadores de registros importados

#### **Treinamento de Modelo**
- **Modelos:** XGBoost (padrão)
- **Monitoramento:** Progress bar e tempo de treinamento
- **Resultados:** Métricas detalhadas

### **3. Análise de Dados**

#### **Filtros**
- **Faixa Etária:** Slider de categorias
- **BMI:** Slider contínuo (10.0-100.0)
- **Sexo:** Dropdown (Todos/Feminino/Masculino)

#### **Visualizações**
- **Distribuição de BMI** por status de diabetes
- **Fatores de risco** - taxas por condição
- **Correlações** entre indicadores (heatmap)
- **Análise demográfica** por faixa etária
- **Violin plots** e histogramas adaptativos

## **Como Executar**

### **Local com Docker:**
```bash
# Executar sistema completo
docker compose up --build -d

# Acesse: http://localhost:8501
```

### **Desenvolvimento:**
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar Streamlit
streamlit run app/main.py --server.port 8501
```

## **Troubleshooting**

### **Problemas Comuns**

#### **API não conecta**
Verificar status na sidebar

#### **Gráficos não aparecem**
```python
# Sempre usar plt.close()
fig, ax = plt.subplots()
st.pyplot(fig)
plt.close(fig)  # não esquecer
```

#### **Upload falha**
- Verificar formato CSV
- Confirmar colunas obrigatórias  
- Checar encoding (UTF-8)

### **🔍 Debug Mode**
```bash
streamlit run app/main.py --logger.level debug
```
---
