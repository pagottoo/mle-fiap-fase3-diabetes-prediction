# Sistema de Predição de Diabetes - Pipeline de Machine Learning

**Pipeline completo de ML com XGBoost, SHAP explicabilidade e treinamento automatizado**

## 🎯 Visão Geral

O módulo ML implementa um pipeline robusto para:
- **Treinamento automatizado** de modelos XGBoost
- **Explicabilidade SHAP** para interpretação de predições
- **Preprocessamento** de dados médicos
- **Validação** com métricas clínicas
- **Persistência de modelos** e metadatos

## Arquitetura do Pipeline

```
Dados BRFSS2015 (69.769 registros)
         ↓
Preprocessamento (train.py)
    ├── Limpeza de dados
    ├── Normalização BMI
    ├── Codificação categórica
    └── Balanceamento classes
         ↓
Treinamento XGBoost (train.py)
    ├── Split treino/teste (80/20)
    ├── Otimização hiperparâmetros
    ├── Validação cruzada
    └── Métricas de avaliação
         ↓
Persistência (artifacts/)
    ├── Modelo treinado (.pkl)
    ├── Preprocessor (.pkl)
    ├── SHAP explainer (.pkl)
    └── Métricas (.json)
         ↓
Predição (predict.py)
    ├── Carregamento modelo
    ├── Preprocessamento input
    ├── Predição probabilística
    └── Explicação SHAP
```

## Estrutura de Arquivos

```
ml/
├──train.py                 # Pipeline de treinamento
├──predict.py               # Sistema de predições
├──explainer.py             # SHAP explicabilidade
├──utils.py                 # Funções utilitárias
├──artifacts/               # Modelos persistidos
│   ├── xgb_model.pkl       # Modelo XGBoost
│   ├── preprocessor.pkl    # Pipeline preprocessamento
│   ├── shap_explainer.pkl  # SHAP explainer
│   └── metrics.json        # Métricas performance
└──README.md                # Esta documentação
```

## Performance do Modelo

### **Métricas Atuais (XGBoost)**
```json
{
  "auc_roc": 0.8242,        // Excelente discriminação
  "accuracy": 0.7481,       // Alta precisão geral
  "f1_score": 0.7612,       // Bom equilíbrio P/R
  "precision": 0.7389,      // Poucos falsos positivos
  "recall": 0.7849,         // Detecta maioria dos casos
  "specificity": 0.7113     // Poucos falsos negativos
}
```

### **Feature Importance (Top 10)**
```
1. BMI (0.234)                    - Fator de risco #1
2. Age (0.198)                    - Idade avançada
3. GenHlth (0.156)                - Saúde geral ruim  
4. HighBP (0.143)                 - Pressão arterial
5. HighChol (0.128)               - Colesterol
6. PhysHlth (0.087)               - Problemas físicos
7. Income (0.076)                 - Fator socioeconômico
8. HeartDiseaseorAttack (0.065)   - Comorbidade cardiovascular
9. PhysActivity (0.058)           - Sedentarismo
10. MentHlth (0.045)              - Saúde mental
```
