"""Feature engineering — reproduit exactement les transformations validées
lors de l'exploration sur Kaggle (notebook 02_feature_engineering).

Ces mêmes transformations doivent être appliquées à l'identique à
l'entraînement et en production (API) : les regrouper ici garantit qu'on ne
les réécrit jamais deux fois différemment par erreur.
"""

import numpy as np
import pandas as pd

from src.config import NIGHT_START_HOUR, NIGHT_END_HOUR


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes Hour, Amount_log et is_night à un DataFrame brut.

    Le DataFrame d'entrée doit contenir au minimum les colonnes `Time` et
    `Amount` (telles que fournies par le dataset Kaggle Credit Card Fraud).
    """
    df = df.copy()

    # Heure de la journée (0-23), à partir du compteur brut en secondes
    df["Hour"] = (df["Time"] // 3600) % 24

    # Montant en échelle log, pour compresser l'asymétrie forte de Amount
    df["Amount_log"] = np.log1p(df["Amount"])

    # Indicateur binaire "transaction de nuit" (taux de fraude ~3.4x plus élevé)
    df["is_night"] = df["Hour"].between(NIGHT_START_HOUR, NIGHT_END_HOUR).astype(int)

    return df
