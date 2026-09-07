"""Chargement du modèle entraîné et prédiction sur de nouvelles transactions.

Reproduit la logique validée à l'étape 5 : le modèle est interrogé via
predict_proba() (probabilité de fraude), puis comparé au seuil de décision
optimisé (0.135) plutôt qu'au seuil par défaut de scikit-learn/XGBoost (0.5).
"""

import pandas as pd
from xgboost import XGBClassifier

from src.config import MODEL_PATH, DECISION_THRESHOLD, FEATURE_COLUMNS
from src.features import build_features

# Chargé une seule fois à l'import du module, pas à chaque appel de fonction
# (charger un modèle depuis le disque a un coût — on ne veut pas le payer à
# chaque transaction scorée, ce qui casserait la contrainte de latence).
#
# Chargement via le format natif XGBoost (pas joblib/pickle) : on recrée un
# classifieur vide, puis on lui fait charger la structure des arbres et les
# attributs sklearn (classes_, etc.) depuis le fichier JSON — stable entre
# versions différentes de la librairie, contrairement à joblib.dump/load.
_model = XGBClassifier()
_model.load_model(MODEL_PATH)


def predict_fraud(transaction: pd.DataFrame) -> pd.DataFrame:
    """Prédit si chaque transaction du DataFrame est frauduleuse.

    `transaction` doit contenir les colonnes brutes attendues par le modèle
    (Time, Amount, V1..V28), avant feature engineering.

    Retourne un DataFrame avec deux colonnes ajoutées :
    - fraud_probability : probabilité prédite (entre 0 et 1)
    - is_fraud : décision finale (0/1), selon le seuil optimisé DECISION_THRESHOLD
    """
    enriched = build_features(transaction)

    # Conversion en numpy (plutôt que de passer le DataFrame directement) :
    # XGBoost revalide les noms de colonnes à chaque appel avec un DataFrame,
    # un chemin ~15x plus lent que l'inférence pure sur cet environnement.
    # reindex garantit l'ordre exact attendu par le modèle avant conversion.
    X = enriched.reindex(columns=FEATURE_COLUMNS).values
    proba = _model.predict_proba(X)[:, 1]

    result = transaction.copy()
    result["fraud_probability"] = proba
    result["is_fraud"] = (proba >= DECISION_THRESHOLD).astype(int)

    return result
