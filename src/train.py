"""Script d'entraînement reproductible.

Reproduit en local, en un seul run, tout le pipeline validé sur Kaggle
(étapes 4-5) : chargement des données -> feature engineering -> split ->
recherche d'hyperparamètres (F2) -> choix du seuil de décision -> sauvegarde
du modèle au format natif XGBoost.

Usage :
    python -m src.train
    python -m src.train --output models/xgb_fraud_model_retrained.json
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score, make_scorer, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import BASE_DIR, MODEL_PATH
from src.features import build_features

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def find_best_threshold(y_true, y_proba):
    """Trouve le seuil qui maximise le F2-score, comme à l'étape 5."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (
        4 * precisions[:-1] + recalls[:-1] + 1e-10
    )
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f2_scores[best_idx]


def main(output_path):
    print(f"Chargement des données : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Recherche d'hyperparamètres (RandomizedSearchCV, scoring=F2)...")
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
        ),
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring=f2_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("Meilleurs hyperparamètres :", search.best_params_)
    print("Meilleur score F2 (cross-validation) :", search.best_score_)

    best_model = search.best_estimator_

    y_pred_default = best_model.predict(X_test)
    print("\n--- Évaluation au seuil par défaut (0.5) ---")
    print(classification_report(y_test, y_pred_default))

    y_proba = best_model.predict_proba(X_test)[:, 1]
    threshold, precision, recall, f2 = find_best_threshold(y_test, y_proba)

    print("\n--- Seuil de décision optimisé (F2) ---")
    print(f"Seuil : {threshold:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F2-score : {f2:.4f}")

    best_model.save_model(output_path)
    print(f"\nModèle sauvegardé : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help="Chemin de sauvegarde du modèle (défaut : chemin de production, config.MODEL_PATH)",
    )
    args = parser.parse_args()
    main(args.output)
