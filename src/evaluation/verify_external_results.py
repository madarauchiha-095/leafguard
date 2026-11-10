"""Audit prediction consistency and recompute metrics for one external test run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)

from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual, expected, tolerance=1e-12):
    return abs(float(actual) - float(expected)) <= tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
    expected_ids = {row["image_id"] for row in payload["eligible_records"]}
    expected_truth = {row["image_id"]: row["true_label"] for row in payload["eligible_records"]}
    prediction_path = args.results_dir / f"{args.dataset_slug}_predictions.csv"
    with prediction_path.open("r", newline="", encoding="utf-8-sig") as stream:
        predictions = list(csv.DictReader(stream))
    grouped = {}
    for row in predictions:
        row["confidence"] = float(row["confidence"])
        row["correct"] = row["correct"].lower() == "true"
        grouped.setdefault(row["model"], []).append(row)

    findings = {
        "manifest_inventory": {
            "expected_unique_image_ids": len(expected_ids),
            "actual_records": len(payload["eligible_records"]),
            "all_unique": len(expected_ids) == len(payload["eligible_records"]),
            "eligible_labels": sorted(set(expected_truth.values())),
        },
        "models": {},
        "same_sample_set_across_models": True,
        "all_metrics_recomputed": True,
    }
    reference_ids = None
    reference_truth = None
    for model_key in MODEL_SPECS:
        rows = grouped.get(model_key, [])
        ids = [row["image_id"] for row in rows]
        id_set = set(ids)
        truths = {row["image_id"]: row["true_label"] for row in rows}
        shared_ok = id_set == expected_ids and len(ids) == len(id_set)
        labels_ok = truths == expected_truth
        if reference_ids is not None:
            shared_ok = shared_ok and id_set == reference_ids
            labels_ok = labels_ok and truths == reference_truth
        findings["same_sample_set_across_models"] &= shared_ok and labels_ok
        reference_ids, reference_truth = id_set, truths

        result_path = args.results_dir / f"{model_key}_{args.dataset_slug}_test_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mapped = result["mapped_classes"]
        true = [row["true_label"] for row in rows]
        pred = [row["predicted_label"] for row in rows]
        recomputed = {
            "accuracy": accuracy_score(true, pred),
            "precision_macro": precision_score(true, pred, labels=mapped, average="macro", zero_division=0),
            "recall_macro": recall_score(true, pred, labels=mapped, average="macro", zero_division=0),
            "f1_macro": f1_score(true, pred, labels=mapped, average="macro", zero_division=0),
            "f1_weighted": f1_score(true, pred, labels=mapped, average="weighted", zero_division=0),
        }
        per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
            true, pred, labels=mapped, zero_division=0,
        )
        per_class_checks = {}
        for index, label in enumerate(mapped):
            reported = result["per_class"][label]
            per_class_checks[label] = {
                "support_match": int(per_class_support[index]) == int(reported["support"]),
                "precision_match": close(per_class_p[index], reported["precision"]),
                "recall_match": close(per_class_r[index], reported["recall"]),
                "f1_match": close(per_class_f1[index], reported["f1-score"]),
            }
        cm = confusion_matrix(true, pred, labels=CLASS_NAMES)
        cm_matches = np.array_equal(cm, np.asarray(result["confusion_matrix"]))
        metric_comparisons = {
            metric: {
                "reported": result["metrics"][metric],
                "recomputed": float(value),
                "match": close(value, result["metrics"][metric]),
            }
            for metric, value in recomputed.items()
        }
        metric_ok = (all(item["match"] for item in metric_comparisons.values())
                     and all(all(check.values()) for check in per_class_checks.values()) and cm_matches)
        findings["all_metrics_recomputed"] &= metric_ok

        checkpoint_path = MODEL_SPECS[model_key]["checkpoint"]
        expected_checkpoint = result.get("checkpoint")
        digest = sha256(checkpoint_path)
        checkpoint_record = checkpoint_manifest["checkpoints"][model_key]
        findings["models"][model_key] = {
            "sample_count": len(rows),
            "identical_inventory_and_labels": bool(shared_ok and labels_ok),
            "no_duplicate_model_image_rows": len(ids) == len(id_set),
            "reported_sample_count_matches": result["metrics"]["total"] == len(rows),
            "metrics": metric_comparisons,
            "per_class_metrics": per_class_checks,
            "confusion_matrix_matches": bool(cm_matches),
            "checkpoint_path_matches_result": checkpoint_path.relative_to(ROOT).as_posix() == Path(expected_checkpoint).as_posix(),
            "checkpoint_sha256": digest,
            "checkpoint_sha256_matches_manifest": digest == checkpoint_record["sha256"],
            "all_10_output_labels_allowed": set(pred).issubset(CLASS_NAMES),
            "unsupported_output_predictions": sum(label not in mapped for label in pred),
            "correctness_flags_match_labels": all(row["correct"] == (row["true_label"] == row["predicted_label"])
                                                   for row in rows),
            "confidence_finite_and_in_range": all(np.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1
                                                    for row in rows),
        }

    statuses = [findings["manifest_inventory"]["all_unique"], findings["same_sample_set_across_models"],
                findings["all_metrics_recomputed"]]
    for model in findings["models"].values():
        statuses.extend([
            model["identical_inventory_and_labels"], model["no_duplicate_model_image_rows"],
            model["reported_sample_count_matches"], model["confusion_matrix_matches"],
            model["checkpoint_path_matches_result"], model["all_10_output_labels_allowed"],
            model["checkpoint_sha256_matches_manifest"],
            model["correctness_flags_match_labels"], model["confidence_finite_and_in_range"],
            *[item["match"] for item in model["metrics"].values()],
            *[check for class_checks in model["per_class_metrics"].values() for check in class_checks.values()],
        ])
    findings["audit_status"] = "VERIFIED" if all(statuses) else "ERROR"
    findings["note"] = "Validation is against the selected mapped sample inventory. It does not independently adjudicate disease labels, source independence, or visual scope."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"{findings['audit_status']}: wrote {args.output}")
    if findings["audit_status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
