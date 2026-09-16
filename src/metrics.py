"""Honest-метрики: Recall@FPR, Stratified K-Fold CV."""
from __future__ import annotations
from typing import Dict, List, Optional
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, probs: np.ndarray, target_fpr: float) -> float:
    """Максимальный recall при FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok[-1]]) if len(ok) > 0 else 0.0


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf_proto,
    target_fprs=(0.05, 0.10, 0.20, 0.30),
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Stratified K-Fold. Для каждой модели возвращает mean/std по AUC, AP, Recall@FPR.
    clf_proto должен иметь .fit, .predict_proba (sklearn-совместимый).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for tr, te in skf.split(X, y):
        clf = clone(clf_proto)
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf.fit(X_tr, y[tr])
        p = clf.predict_proba(X_te)[:, 1]
        row = {
            "auc": float(roc_auc_score(y[te], p)),
            "ap": float(average_precision_score(y[te], p)),
        }
        for f in target_fprs:
            row[f"r{f}"] = recall_at_fpr(y[te], p, f)
        rows.append(row)

    df = pd.DataFrame(rows)
    return {"mean": df.mean(numeric_only=True).to_dict(),
            "std": df.std(numeric_only=True).to_dict(),
            "per_fold": rows}


def run_cv_all(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: Dict[str, object],
    target_fprs=(0.05, 0.10, 0.20, 0.30),
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Прогоняет CV для набора моделей. Возвращает dict {name: cv_result}."""
    out: Dict[str, Dict] = {}
    for name, clf in classifiers.items():
        if verbose:
            print(f"\n▶ {name}")
        t0 = time.time()
        res = run_cv(X, y, clf, target_fprs, n_splits, seed)
        res["total_time_s"] = round(time.time() - t0, 2)
        out[name] = res
        if verbose:
            m, s = res["mean"], res["std"]
            print(f"   AUC={m['auc']:.3f}±{s['auc']:.3f}  "
                  f"AP={m['ap']:.3f}±{s['ap']:.3f}  "
                  f"R@20%={m['r0.2']:.3f}  ({res['total_time_s']}s)")
    return out