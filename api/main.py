"""API de scoring de fraude — expose le modèle de prédiction via HTTP.

Lancement local :
    uvicorn api.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from api.cache import get_cached, set_cached
from api.schemas import PredictionResponse, Transaction
from src.predict import predict_fraud

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Scoring de fraude en temps réel sur des transactions bancaires",
    version="1.0.0",
)

# Métriques HTTP automatiques (nombre de requêtes, latence, requêtes en
# cours), publiées au format Prometheus sur l'endpoint /metrics.
Instrumentator().instrument(app).expose(app)

# Métriques métier, incrémentées manuellement à chaque scoring.
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total", "Nombre total de transactions scorées"
)
FRAUD_FLAGGED_TOTAL = Counter(
    "fraud_flagged_total", "Nombre de transactions classées comme fraude"
)
FRAUD_PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution des probabilités de fraude prédites",
    buckets=(0.01, 0.05, 0.135, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
)
CACHE_HITS_TOTAL = Counter(
    "fraud_cache_hits_total", "Nombre de requêtes servies depuis le cache Redis"
)


@app.get("/health")
def health():
    """Vérifie que l'API est en vie (sondes de monitoring et d'orchestration)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Score une transaction unique et renvoie la décision de fraude."""
    # FastAPI a déjà validé `transaction` grâce au schéma Pydantic — on est
    # certain, à ce stade, que tous les champs attendus sont présents et
    # du bon type.
    payload = transaction.model_dump()

    # Transaction déjà scorée récemment : on renvoie le résultat mémorisé.
    cached = get_cached(payload)
    if cached is not None:
        CACHE_HITS_TOTAL.inc()
        return PredictionResponse(**cached)

    result = predict_fraud(pd.DataFrame([payload]))
    proba = float(result["fraud_probability"].iloc[0])
    is_fraud = bool(result["is_fraud"].iloc[0])

    PREDICTIONS_TOTAL.inc()
    FRAUD_PROBABILITY.observe(proba)
    if is_fraud:
        FRAUD_FLAGGED_TOTAL.inc()

    response = {"fraud_probability": proba, "is_fraud": is_fraud}
    set_cached(payload, response)
    return PredictionResponse(**response)
