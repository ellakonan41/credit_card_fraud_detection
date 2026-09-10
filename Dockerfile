# Image de l'API de scoring de fraude.

FROM python:3.12-slim

WORKDIR /app

# Les dépendances d'abord : cette couche Docker reste en cache tant que
# requirements.txt ne change pas (rebuilds plus rapides).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif + modèle entraîné.
COPY src/ ./src/
COPY api/ ./api/
COPY models/xgb_fraud_model.json ./models/xgb_fraud_model.json

EXPOSE 8000

# Forme shell (et non liste) pour que ${PORT} soit interprété : Render fournit
# son propre port via cette variable ; en local, on retombe sur 8000.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
