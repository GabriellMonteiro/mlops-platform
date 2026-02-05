# MLOps com API de Treinamento

## Subir ambiente
docker compose -f docker-compose.dev.yml up --build

## Disparar treino via API
POST http://localhost:8000/train

Swagger:
http://localhost:8000/docs

MLflow:
http://localhost:5000
