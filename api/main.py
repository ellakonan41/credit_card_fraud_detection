"""API de scoring de fraude — expose predict_fraud() (src/) via HTTP.

Lancement local :
    uvicorn api.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI

from api.schemas import PredictionResponse, Transaction
from src.predict import predict_fraud

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Scoring de fraude en temps réel sur des transactions bancaires",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Endpoint de vérification que l'API est en vie (utile pour le
    monitoring à l'étape 8, et pour Docker à l'étape 9)."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Score une transaction unique et renvoie la décision de fraude."""
    # FastAPI a déjà validé `transaction` grâce au schéma Pydantic — on est
    # certain, à ce stade, que tous les champs attendus sont présents et
    # du bon type.
    df = pd.DataFrame([transaction.model_dump()])

    result = predict_fraud(df)

    return PredictionResponse(
        fraud_probability=float(result["fraud_probability"].iloc[0]),
        is_fraud=bool(result["is_fraud"].iloc[0]),
    )
