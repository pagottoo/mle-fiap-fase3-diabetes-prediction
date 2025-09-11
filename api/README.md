# Sistema de Predição de Diabetes - API REST com FastAPI

**Backend RESTful para predições de diabetes, gerenciamento de dados e treinamento de modelos**

## Visão Geral

Esta API fornece endpoints para:
- **Predições em tempo real** de risco de diabetes
- **Gerenciamento de dados** (upload CSV/Datasets, cadastro manual)
- **Treinamento de modelos** com dados customizados
- **Métricas de performance** e monitoramento simples
- **Health checks** e status do sistema

## Links Uteis:

- **URL Base:** http://localhost:8000
- **Docs:** http://localhost:8000/docs
- **Redoc:** http://localhost:8000/redoc
- **Status:** http://localhost:8000/status

## Estrutura:

```
api/
├── main.py           # FastAPI app e rotas principais
├── models.py         # Modelos SQLAlchemy
├── schemas.py        # Schemas Pydantic
├── db.py             # Configuração banco
├── config.py         # Configurações ambiente
└── README.md         # Esta documentação

Endpoints:
├── POST /predict       # Predição de diabetes
├── GET  /records       # Listar dados
├── POST /upload-csv    # Upload de dados
├── POST /train         # Treinar modelo
├── GET  /metrics       # Métricas modelo
└── GET  /status        # Health check
```

## Detalhamento dos endpoints:

### **POST /predict**
Realiza predição de risco de diabetes para um paciente.

### **GET /records**
Lista registros de dados com paginação.

**Parâmetros:**
- `limit`: Número máximo de registros (padrão: 100)
- `offset`: Número de registros para pular (padrão: 0)

### **POST /train**
Treina novo modelo com dados do banco.

### **GET /metrics**
Obtém métricas do modelo atual.

### **GET /status**
Verifica status geral do sistema.

Mais informações do Postman da API.

## Configuração

### **Variáveis de Ambiente**

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/diabetes_db

# ML Settings
MODEL_PATH=/app/ml/artifacts/
ENABLE_SHAP=true
MAX_SHAP_SAMPLES=1000

# API Settings
API_DEBUG=false
API_WORKERS=4
CORS_ORIGINS=http://localhost:8501
```

## Desenvolvimento

### **Executar Localmente**
```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar banco
export DATABASE_URL=postgresql://user:pass@localhost:5432/diabetes_db

# Executar API
uvicorn api.main:app --reload --port 8000
```

### **Teste Manual com curl**
```bash
# Health check
curl http://localhost:8000/status

# Predição
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "high_bp": 1, "high_chol": 1, "bmi": 28.5,
    "age": 8, "sex": 1, "education": 6
  }'
```

## Throubloshooting

### **Problemas Comuns**

#### **Erro de Conexão com Banco**
```bash
# Verificar se o PostgreSQL está rodando
docker ps | grep postgres

# Testar conexão
psql postgresql://user:pass@localhost:5432/diabetes_db
```

#### **Modelo Não Carregado**
```bash
# Verificar se os arquivos do modelo estão no diretório esperado
ls -la ml/artifacts/

# Logs da API
docker logs diabetes_health_api --tail 50
```

### Outros Problemas Comuns

1. **API não inicia**
   - Verificar dependências: `make install`
   - Verificar porta em uso: `lsof -i :8000`

2. **Predições falham**
   - Treinar modelo: `make train`
   - Verificar dados: validação Pydantic mostrará erros

3. **Banco de dados indisponível**
   - Verificar PostgreSQL rodando
   - Verificar configurações em `api/config.py`

4. **Performance lenta**
   - Verificar logs para gargalos
   - Considerar cache Redis
   - Aumentar workers: `--workers 4`

### Debug Mode
```bash
# Adicionar debug logs
export LOG_LEVEL=DEBUG

# Rodar em modo debug
uvicorn api.main:app --reload --log-level debug
```
