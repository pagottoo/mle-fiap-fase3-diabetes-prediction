# Dados do Sistema de Predição de Diabetes

Esta pasta contém os datasets necessários para o funcionamento do sistema de predição de diabetes.

## ⚠️ **Importante** ⚠️

Os arquivos de dados **NÃO estão incluídos no repositório** devido ao tamanho e políticas de privacidade. Você deve baixá-los manualmente seguindo as instruções abaixo.

## **Como Obter os Dados**

### **Opção 1: Download Automático (Recomendado)**

Execute o script de download automático:

```bash
# Na raiz do projeto
python -m scripts.download_data
```
OBS: para o script funcionar você precisa ter o cadastro no Kaggle e exportar sua chave de API.

### **Opção 2: Download Manual**

1. **Acesse o Kaggle Dataset:**
   - URL: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
   - Título: "Diabetes Health Indicators Dataset"

2. **Faça o Download:**
   - Baixe o arquivo `diabetes-health-indicators-dataset.zip`
   - Extraia o conteúdo na pasta `data/`

3. **Arquivos Necessários:**
   ```
   data/
   ├── diabetes.csv                                                 # Dataset principal
   ├── diabetes_binary_health_indicators_BRFSS2015.csv              # Dataset binário
   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv    # Dataset balanceado
   └── diabetes_012_health_indicators_BRFSS2015.csv                 # Dataset com 3 classes
   ```

## **Descrição dos Datasets**

### **1. diabetes.csv** **(Principal)**
- **Descrição**: Dataset principal usado pelo sistema
- **Registros**: 253.680 observações
- **Features**: 21 indicadores de saúde
- **Target**: Variável binária (0 = Sem diabetes, 1 = Com diabetes)
- **Fonte**: CDC BRFSS 2015

### **2. diabetes_binary_health_indicators_BRFSS2015.csv**
- **Descrição**: Dataset binário completo
- **Uso**: Treinamento de modelos binários
- **Target**: 0/1 (Sem/Com diabetes)

### **3. diabetes_binary_5050split_health_indicators_BRFSS2015.csv**
- **Descrição**: Dataset balanceado 50/50
- **Registros**: 69.769 observações
- **Uso**: Treinamento com classes balanceadas
- **Vantagem**: Evita bias de classe majoritária

### **4. diabetes_012_health_indicators_BRFSS2015.csv**
- **Descrição**: Dataset com 3 classes
- **Target**: 0 = Sem diabetes, 1 = Pré-diabetes, 2 = Diabetes
- **Uso**: Análises mais granulares

## **Estrutura Final Esperada**

Após o download, sua pasta `data/` deve estar assim:

```
data/
├── README.md                                             # Este arquivo que você lê
├── .gitkeep                                              # arquivo do git
├── diabetes.csv                                          # esse é o monstro
├── diabetes_binary_health_indicators_BRFSS2015.csv
├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
├── diabetes_012_health_indicators_BRFSS2015.csv
└── diabetes-health-indicators-dataset.zip                # Arquivo original (opcional)
```

**Com os dados configurados corretamente, o sistema estará pronto para usar com `docker compose up --build -d`!**

Volte para o README principal, mas basicamente o docker compose up já vai deixar tudo pronto!
