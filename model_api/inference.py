"""
Inference utilities — CatBoost only
-----------------------------------
* Загружает модель из  models/catboost_model.cbm
* Выравнивает входной DataFrame по model.feature_names_
* Добавляет отсутствующие колонки с нейтральным значением
* Возвращает pd.Series с вероятностями конверсии
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# ---------------------------------------------------------------------
_MODEL_PATH = Path("models") / "catboost_model.cbm"


def _load_cat() -> tuple[CatBoostClassifier, list[int]]:
    """Загружает CatBoost-модель и индексы категориальных признаков."""
    model = CatBoostClassifier()
    model.load_model(_MODEL_PATH)

    json_path = Path(__file__).parent / "cat_features.json"
    if json_path.exists():
        cat_feats: list[int] = json.loads(json_path.read_text())
    else:
        cat_feats = model.get_cat_feature_indices()  # fallback

    return model, cat_feats


# ---------------------------------------------------------------------
def predict(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Выход preprocess_from_raw() (без 'target').

    Returns
    -------
    pd.Series
        Вероятность положительного класса для каждой строки.
    """
    model, cat_idx = _load_cat()
    expected_cols = model.feature_names_
    
    # --- добавляем недостающие признаки ---
    for col in expected_cols:
        if col not in df.columns:
            if col.endswith(("_0", "_1")) or col.startswith("is_"):
                df[col] = 0                       # бинарные
            else:
                df[col] = np.nan                  # остальные
    # --- упорядочиваем ---
    df = df[expected_cols]
    
    # --- обработка категориальных: NaN -> "" , тип string ---
    for i in cat_idx:
        col = expected_cols[i]
        df[col] = df[col].astype("string").fillna("")
        
    pool = Pool(df, cat_features=cat_idx)
    prob = model.predict_proba(pool)[:, 1]
    return pd.Series((prob >= threshold).astype(int), index=df.index)
