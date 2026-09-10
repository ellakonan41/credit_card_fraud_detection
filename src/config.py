"""Configuration centralisée du projet (chemins, hyperparamètres de décision).

Regrouper ces valeurs ici évite d'avoir des "nombres magiques" (0.135, un
chemin de fichier...) dispersés et dupliqués dans plusieurs fichiers du code.
"""

from pathlib import Path

# Racine du projet = un dossier au-dessus de src/ (là où se trouve ce fichier)
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemin vers le modèle XGBoost entraîné.
# Format JSON natif XGBoost (pas pickle/joblib) : stable entre versions
# différentes de la librairie, contrairement à joblib.dump/load.
MODEL_PATH = BASE_DIR / "models" / "xgb_fraud_model.json"

# Seuil de décision, optimisé sur le F2-score (au lieu du 0.5 par défaut).
DECISION_THRESHOLD = 0.135

# Bornes horaires pour la feature "is_night".
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6

# Ordre exact des colonnes attendu par le modèle (celui de l'entraînement :
# colonnes brutes, puis features dérivées). Le fixer explicitement évite de
# dépendre implicitement de l'ordre d'insertion des colonnes dans
# build_features().
FEATURE_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour", "Amount_log", "is_night"]
)
