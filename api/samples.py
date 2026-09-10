"""Transactions d'exemple pour la démo (`GET /predict/sample`).

Ces transactions sont tirées du **jeu de test** — le modèle ne les a jamais
vues à l'entraînement. Chaque ligne garde sa vraie étiquette (`Class`), ce
qui permet à la démo d'afficher si la prédiction est correcte.
"""

from pathlib import Path

import pandas as pd

_PATH = Path(__file__).parent / "sample_transactions.csv"
_df = pd.read_csv(_PATH)

_RAW_COLUMNS = [c for c in _df.columns if c != "Class"]


def get_sample(fraud: bool | None = None) -> tuple[dict, bool]:
    """Tire une transaction au hasard et renvoie (transaction brute, vraie étiquette).

    fraud=True   -> uniquement de vraies fraudes
    fraud=False  -> uniquement de vraies transactions normales
    fraud=None   -> n'importe laquelle
    """
    pool = _df
    if fraud is True:
        pool = _df[_df["Class"] == 1]
    elif fraud is False:
        pool = _df[_df["Class"] == 0]

    row = pool.sample(1).iloc[0]
    transaction = {col: float(row[col]) for col in _RAW_COLUMNS}
    actual_is_fraud = bool(row["Class"])
    return transaction, actual_is_fraud
