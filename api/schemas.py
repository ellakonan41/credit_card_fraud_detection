"""Schémas Pydantic : forme exacte des requêtes/réponses de l'API.

FastAPI utilise ces classes pour valider automatiquement chaque requête
entrante (types corrects, champs obligatoires présents) avant même que notre
code ne s'exécute — une requête mal formée est rejetée avec une erreur 422
claire, sans qu'on ait à écrire de vérifications manuelles.
"""

from pydantic import BaseModel


class Transaction(BaseModel):
    """Une transaction brute, telle qu'attendue par le modèle (avant
    feature engineering, qui est fait automatiquement côté serveur)."""

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    """Résultat renvoyé par l'API pour une transaction scorée."""

    fraud_probability: float
    is_fraud: bool


class SampleResponse(BaseModel):
    """Résultat de /predict/sample : une transaction de test, sa prédiction,
    sa vraie étiquette, et si le modèle a vu juste."""

    transaction: Transaction
    prediction: PredictionResponse
    actual_is_fraud: bool
    correct: bool
