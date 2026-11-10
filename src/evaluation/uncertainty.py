"""Stratified image-bootstrap intervals from saved confusion matrices.

This conditions on observed class supports and resamples outcomes within each
true class. It does not estimate uncertainty from dataset/source construction.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("mobilenet", "mobilenet_clahe", "deit_tiny", "hybrid")
MAPPED = (0, 2, 3, 4, 5, 6, 7, 8, 9)
REPLICATES = 10_000
SEED = 20260924


def stratified_bootstrap(matrix, labels, rng, replicates=REPLICATES):
    """Percentile intervals for accuracy and macro-F1, preserving class support."""
    cm = np.asarray(matrix, dtype=np.int64)[np.asarray(labels, dtype=int), :]
    supports = cm.sum(axis=1)
    if np.any(supports == 0):
        raise ValueError("Cannot bootstrap a class with zero support")
    sampled_rows = np.stack([
        rng.multinomial(int(n), row / n, size=replicates)
        for row, n in zip(cm, supports)
    ], axis=1)
    tp = sampled_rows[:, np.arange(len(labels)), np.asarray(labels)]
    predicted = sampled_rows.sum(axis=1)
    precision = tp / np.maximum(predicted[:, np.asarray(labels)], 1)
    recall = tp / supports[None, :]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    macro_f1 = f1.mean(axis=1)
    accuracy = tp.sum(axis=1) / supports.sum()
    return {
        "accuracy_95_percentile_ci": np.percentile(accuracy, [2.5, 97.5]).round(6).tolist(),
        "macro_f1_95_percentile_ci": np.percentile(macro_f1, [2.5, 97.5]).round(6).tolist(),
        "resamples": replicates,
        "resampling": "within each true class, with observed class support fixed",
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {
        "method": "Stratified image bootstrap from confusion-matrix counts; percentile intervals.",
        "seed": SEED,
        "replicates": REPLICATES,
        "interpretation_limit": "Intervals condition on the observed samples, class mapping, and class supports; they do not capture dataset/source selection uncertainty. They are descriptive and are not a paired test of model differences.",
        "models": {},
    }
    for model_key in MODELS:
        pv_path = ROOT / "results/audit" / f"{model_key}_plantvillage_verified.json"
        ext_path = ROOT / "results/domain_shift" / f"{model_key}_plantdoc_test_results.json"
        pv = json.loads(pv_path.read_text(encoding="utf-8"))
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        pv_cm = pv["confusion_matrix"]
        ext_cm = ext["confusion_matrix"]
        results["models"][model_key] = {
            "plantvillage_full_10_classes": stratified_bootstrap(pv_cm, list(range(10)), rng),
            "plantvillage_mapped_9_classes": stratified_bootstrap(pv_cm, MAPPED, rng),
            "plantdoc_mapped_9_classes": stratified_bootstrap(ext_cm, MAPPED, rng),
        }
    out = ROOT / "results/audit/stratified_bootstrap_intervals.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved stratified bootstrap intervals to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
