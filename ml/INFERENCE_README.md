# Módulo de Inferência (ml/infer.py)

## Visão Geral

O módulo de inferência (`ml/infer.py`) é responsável por carregar o modelo treinado e fazer predições sobre novos dados, incluindo explicações SHAP para interpretar as decisões do modelo.

## Funcionalidades Principais

### 1. Carregamento do Pipeline
- **Função**: `load_pipeline(model_path)`
- **Descrição**: Carrega o pipeline ML salvo durante o treinamento
- **Cache**: Mantém o pipeline em memória para evitar recarregamentos desnecessários
- **Validação**: Verifica se o objeto carregado possui os métodos necessários

### 2. Validação de Entrada
- **Função**: `validate_payload(payload)`
- **Descrição**: Valida e limpa dados de entrada para predição
- **Validações**:
  - Presença de todas as 13 features obrigatórias
  - Tipos de dados corretos (int/float conforme esperado)
  - Ranges válidos para cada feature
  - Tratamento de features extras (warning, não erro)

### 3. Predição Individual
- **Função**: `predict_one(payload, threshold=0.5)`
- **Entrada**: Dicionário com 13 features do paciente
- **Saída**: Dicionário com:
  - `probability`: Probabilidade de doença cardíaca (0-1)
  - `class`: Classe predita (0=sem risco, 1=com risco)
  - `threshold`: Limiar usado para classificação
  - `confidence`: Confiança da predição (probabilidade máxima)
  - `risk_level`: Categoria de risco ("Low", "Medium", "High")

### 4. Explicações SHAP - Não implementado completamente!
- **Função**: `explain_one(payload, top_k=3)`
- **Descrição**: Gera explicações locais usando SHAP
- **Entrada**: Mesmo payload da predição
- **Saída**: Lista de features mais importantes ordenadas por impacto
- **Estrutura da explicação**:
  ```python
  [
    {
      'feature': 'nome_da_feature',
      'shap_value': 0.234,  # Contribuição SHAP
      'abs_shap_value': 0.234  # Valor absoluto para ordenação
    },
    ...
  ]
  ```

## Features Esperadas

O modelo espera exatamente 13 features nesta ordem:

1. **age**: Idade (1-120 anos)
2. **sex**: Sexo (0=feminino, 1=masculino)
3. **cp**: Tipo de dor no peito (0-3)
4. **trestbps**: Pressão arterial em repouso (80-250 mmHg)
5. **chol**: Colesterol sérico (100-600 mg/dl)
6. **fbs**: Açúcar no sangue em jejum > 120 mg/dl (0=falso, 1=verdadeiro)
7. **restecg**: Resultados ECG em repouso (0-2)
8. **thalach**: Frequência cardíaca máxima alcançada (60-220 bpm)
9. **exang**: Angina induzida por exercício (0=não, 1=sim)
10. **oldpeak**: Depressão ST induzida por exercício (0-10)
11. **slope**: Inclinação do segmento ST no pico do exercício (0-2)
12. **ca**: Número de vasos principais coloridos por fluoroscopia (0-4)
13. **thal**: Talassemia (0-3)

## Categorização de Risco

O módulo categoriza automaticamente o risco baseado na probabilidade:

- **Low Risk**: probabilidade < 0.3
- **Medium Risk**: 0.3 ≤ probabilidade < 0.7  
- **High Risk**: probabilidade ≥ 0.7

## Tratamento de Erros

### Exceções Customizadas
- **ValidationError**: Dados de entrada inválidos
- **InferenceError**: Falhas no processo de inferência

### Erros Tratados
- Modelo não encontrado
- Arquivo corrompido
- SHAP indisponível
- Features faltando ou inválidas
- Tipos de dados incorretos
- Values fora do range esperado

## Dependências

### Obrigatórias
- `pandas`: Manipulação de DataFrames
- `numpy`: Operações numéricas
- `joblib`: Carregamento do modelo
- `scikit-learn`: Pipeline ML

### Opcionais - Não implementado completamente!
- `shap`: Explicações do modelo

## Integração

### Módulo Rest API
O módulo foi projetado para integração direta com endpoints REST:

```python
@app.post("/predict")
def predict_endpoint(patient: PatientSchema):
    try:
        result = predict_one(patient.dict())
        return result
    except ValidationError as e:
        raise HTTPException(400, str(e))
    except InferenceError as e:
        raise HTTPException(500, str(e))
```

### Streamlit Dashboard
Ideal para interfaces interativas:

```python
# Formulário para entrada de dados
patient_data = get_patient_form()

if st.button("Predict"):
    result = predict_and_explain(patient_data)
    
    # Mostrar resultado
    st.metric("Risk Level", result['risk_level'])
    st.metric("Probability", f"{result['probability']:.1%}")
    
    # Mostrar explicações
    for exp in result['explanations']:
        st.bar_chart([exp['shap_value']])
```

## Testes

### Cobertura de Testes (`tests/test_infer.py`)
- Validação de payload (casos válidos e inválidos)
- Carregamento de pipeline (sucesso e falhas)
- Predições individuais
- Explicações SHAP (não implementado completamente)
- Funções combinadas
- Tratamento de erros
- Casos extremos

### Execução dos Testes
```bash
# Testes específicos do módulo
make test-infer

# Todos os testes
make test
