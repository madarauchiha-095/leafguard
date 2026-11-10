"""Audit existing LeafGuard artifacts without retraining any checkpoint."""
import argparse
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision.datasets import ImageFolder

from src.data.prepare_plantdoc import LABEL_TO_PLANTVILLAGE
from src.explainability.gradcampp import explain_pil
from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT, build_model, make_transform
from src.security.integrity import sha256_file, verify_checkpoint

MAPPED_IDS = (0, 2, 3, 4, 5, 6, 7, 8, 9)
PV_COUNTS = {"train": 7647, "val": 845, "test": 2054}


def status_record(status, detail, **extra):
    return {"status": status, "detail": detail, **extra}


def digest(path):
    return sha256_file(Path(path))


def metrics_from_cm(matrix, class_ids):
    cm = np.asarray(matrix, dtype=np.int64)
    class_ids = list(class_ids)
    tp = cm[np.asarray(class_ids), np.asarray(class_ids)]
    actual = cm[np.asarray(class_ids), :].sum(axis=1)
    predicted = cm[:, np.asarray(class_ids)].sum(axis=0)
    support = int(actual.sum())
    p = np.divide(tp, predicted, out=np.zeros(len(class_ids)), where=predicted > 0)
    r = np.divide(tp, actual, out=np.zeros(len(class_ids)), where=actual > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros(len(class_ids)), where=(p+r) > 0)
    return {
        "accuracy": float(tp.sum() / support),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "total": support,
    }


def bootstrap_reference(matrix, class_ids, rng, reps=10_000):
    cm = np.asarray(matrix, dtype=np.int64)[np.asarray(class_ids), :]
    supports = cm.sum(axis=1)
    samples = np.stack([rng.multinomial(int(n), row/n, size=reps)
                        for row, n in zip(cm, supports)], axis=1)
    tp = samples[:, np.arange(len(class_ids)), np.asarray(class_ids)]
    predicted = samples.sum(axis=1)[:, np.asarray(class_ids)]
    p = tp / np.maximum(predicted, 1)
    r = tp / supports[None, :]
    f = 2*p*r / np.maximum(p+r, 1e-15)
    return {
        "accuracy_95_percentile_ci": np.percentile(tp.sum(axis=1)/supports.sum(), [2.5, 97.5]).round(6).tolist(),
        "macro_f1_95_percentile_ci": np.percentile(f.mean(axis=1), [2.5, 97.5]).round(6).tolist(),
    }


def measure_latency(model, device, warmup=8, repeats=3, iterations=60):
    x = torch.zeros(1, 3, 224, 224, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        trials = []
        for _ in range(repeats):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start = time.perf_counter()
            for _ in range(iterations):
                model(x)
            if device.type == "cuda":
                end_event.record()
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000
            trials.append(elapsed_ms / iterations)
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", action="store_true", help="Record pre-correction audit snapshot")
    args = parser.parse_args()
    audit_dir = ROOT / "results/audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "project": "LeafGuard",
        "audit_scope": "Phase 1 reproducibility audit; no training performed.",
        "findings": {},
        "models": {},
        "dataset_overlap": {},
        "independent_metric_recalculation": {},
        "saved_paper_value_check": {},
    }

    # Dataset inventory: ImageFolder labels come from the directory names and order.
    split_inventory = {}
    for split, expected_count in PV_COUNTS.items():
        folder = ImageFolder(ROOT / "data/processed" / split)
        counts = {name: 0 for name in folder.classes}
        for _, label in folder.samples:
            counts[folder.classes[label]] += 1
        checks = {
            "count": status_record("VERIFIED" if len(folder) == expected_count else "ERROR",
                                    f"Found {len(folder)} images; expected {expected_count}.",
                                    observed=len(folder), expected=expected_count),
            "class_folders": status_record("VERIFIED" if folder.classes == CLASS_NAMES else "ERROR",
                                            "ImageFolder class order compared to the model's fixed output order.",
                                            observed=folder.classes, expected=CLASS_NAMES),
        }
        if split == "test":
            checks["per_class_support"] = status_record(
                "VERIFIED" if sum(counts.values()) == 2054 and all(counts.values()) else "ERROR",
                "Counts are folder-label supports; semantic image-label correctness was not independently adjudicated.",
                counts=counts,
            )
            checks["semantic_label_review"] = status_record(
                "NOT TESTED", "Folder labels were audited mechanically; no independent agronomist relabeling was performed."
            )
        split_inventory[split] = checks
    audit["findings"]["plantvillage_splits"] = split_inventory

    training_seed_audit = {}
    for key, filename in {
        "mobilenet": "train_mobilenet.py",
        "mobilenet_clahe": "train_mobilenet_clahe.py",
        "deit_tiny": "train_deit_tiny.py",
        "hybrid": "train_hybrid.py",
    }.items():
        source = (ROOT / "src/training" / filename).read_text(encoding="utf-8")
        seed_match = re.search(r"^SEED\s*=\s*(\d+)\s*$", source, flags=re.MULTILINE)
        uses_torch_seed = "torch.manual_seed(SEED)" in source
        uses_cuda_seed = "torch.cuda.manual_seed_all(SEED)" in source
        complete = bool(seed_match and uses_torch_seed and uses_cuda_seed)
        training_seed_audit[key] = {
            "seed_value": int(seed_match.group(1)) if seed_match else None,
            "python_or_numpy_seed": "not present in inspected script" if "random.seed" not in source and "np.random.seed" not in source else "present",
            "torch_seed_set": uses_torch_seed,
            "cuda_seed_set": uses_cuda_seed,
            "seeded_in_original_training_script": complete,
        }
    audit["findings"]["original_training_seeds"] = status_record(
        "VERIFIED" if all(x["seeded_in_original_training_script"] for x in training_seed_audit.values()) else "WARNING",
        "Training-script seed configuration inspected without launching training. Unseeded legacy runs cannot be reconstructed from checkpoint files alone.",
        per_model=training_seed_audit)

    # Verify all external image identities against both manifests and prepared files.
    prep_path = ROOT / "data/external/plantdoc_prepared/preparation_manifest.json"
    source_path = ROOT / "data/external/plantdoc_source_subset/source_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_by_path = {row["source_path"]: row for row in source["files"]}
    ext = ImageFolder(ROOT / "data/external/plantdoc_prepared/test")
    ext_counts = {name: sum(1 for _, y in ext.samples if ext.classes[y] == name) for name in ext.classes}
    provenance_missing, provenance_hash_mismatch, mapping_mismatch = [], [], []
    ext_hashes = {}
    for entry in prep["manifest"]:
        src_row = source_by_path.get(entry["source_path"])
        out_path = ROOT / "data/external/plantdoc_prepared" / entry["prepared_path"]
        if src_row is None:
            provenance_missing.append(entry["source_path"])
        elif digest(out_path) != src_row["sha256"]:
            provenance_hash_mismatch.append(entry["source_path"])
        normalized = " ".join(entry["source_class"].replace("_", " ").lower().split())
        mapped = LABEL_TO_PLANTVILLAGE.get(normalized)
        if mapped != entry["plantvillage_class"]:
            mapping_mismatch.append(entry["source_path"])
        ext_hashes.setdefault(digest(out_path), []).append(str(out_path))
    ext_count_ok = len(ext) == 83 and prep.get("source_test_images_mapped") == 83 and source.get("image_count") == 83
    all_source_paths_test = all(item.get("source_path", "").replace("\\", "/").startswith("test/")
                                for item in prep["manifest"]) and all(item.get("source_path", "").replace("\\", "/").startswith("test/")
                                                                           for item in source["files"])
    audit["findings"]["plantdoc_inventory"] = {
        "sample_count": status_record("VERIFIED" if ext_count_ok else "ERROR",
                                      "Prepared ImageFolder count and both source manifests agree.",
                                      prepared=len(ext), preparation_manifest=prep.get("source_test_images_mapped"),
                                      source_manifest=source.get("image_count")),
        "source_split": status_record("VERIFIED" if source.get("source_split") == "test only" else "ERROR",
                                       "Materialization manifest split declaration.", split=source.get("source_split"),
                                       pinned_commit=source.get("source_commit")),
        "per_class_counts": status_record("VERIFIED" if ext_counts == prep["mapped_class_counts"] else "ERROR",
                                          "Prepared folder counts compared with preparation manifest.",
                                          observed=ext_counts, manifest=prep["mapped_class_counts"]),
        "provenance_hashes": status_record("VERIFIED" if not provenance_missing and not provenance_hash_mismatch else "ERROR",
                                           "All prepared images compared byte-for-byte with pinned source SHA-256 entries.",
                                           missing=len(provenance_missing), mismatched=len(provenance_hash_mismatch)),
        "label_mapping_code_agreement": status_record("VERIFIED" if not mapping_mismatch else "ERROR",
                                                      "Manifest mappings compared with the current explicit label map.",
                                                      mismatches=mapping_mismatch),
        "semantic_label_review": status_record("WARNING",
                                                "Mapping rules are explicit and reproducible, but PlantDoc labels were not independently expert-adjudicated."),
        "plantdoc_training_images_used": status_record("VERIFIED" if source.get("source_split") == "test only" and all_source_paths_test else "ERROR",
                                                        "The source split declaration and every recorded source path must be test-only.",
                                                        all_paths_test_split=all_source_paths_test),
        "duplicate_within_external": status_record("VERIFIED" if all(len(v) == 1 for v in ext_hashes.values()) else "ERROR",
                                                    "Exact bytewise SHA-256 duplicate scan within mapped PlantDoc subset.",
                                                    exact_duplicate_groups=sum(len(v)>1 for v in ext_hashes.values())),
    }

    pv_test = ImageFolder(ROOT / "data/processed/test")
    pv_hashes = {digest(path): path for path, _ in pv_test.samples}
    exact_cross = [(pv_hashes[h], paths[0]) for h, paths in ext_hashes.items() if h in pv_hashes]
    audit["dataset_overlap"] = {
        "exact_duplicate_scan": status_record("VERIFIED" if not exact_cross else "ERROR",
                                               "SHA-256 comparison across all 2,054 PlantVillage test and 83 mapped PlantDoc images.",
                                               exact_duplicate_pairs=len(exact_cross)),
        "near_duplicate_scan": status_record("NOT TESTED", "No perceptual-hash/manual near-duplicate review was performed."),
    }

    # The external evaluator must create checkpoint-specific transforms and use a
    # single identical ordered sample inventory for all model conditions.
    runner_source = (ROOT / "src/evaluation/domain_shift.py").read_text(encoding="utf-8")
    clahe_transform_ok = ("make_transform(spec[\"clahe\"])" in runner_source or
                          "make_transform(MODEL_SPECS[key][\"clahe\"])" in runner_source)
    runner_uses_single_imagefolder = "inventory = ImageFolder(args.dataset)" in runner_source and "dataset.samples != expected_samples" in runner_source
    pv_runner_source = (ROOT / "src/evaluation/finalize_results.py").read_text(encoding="utf-8")
    pv_transform_ok = 'transform=make_transform(spec["clahe"])' in pv_runner_source
    audit["findings"]["external_preprocessing_and_pairing"] = {
        "checkpoint_specific_clahe": status_record(
            "VERIFIED" if clahe_transform_ok else "ERROR",
            "PlantDoc evaluator must apply each MODEL_SPECS CLAHE flag. False means the CLAHE checkpoint is receiving baseline preprocessing.",
            clahe_model_requires=bool(MODEL_SPECS["mobilenet_clahe"]["clahe"]),
            evaluator_uses_spec_transform=clahe_transform_ok,
        ),
        "same_external_samples": status_record("VERIFIED" if runner_uses_single_imagefolder else "WARNING",
                                               "Runner compares each per-model ImageFolder sample list to one shared source inventory.",
                                               one_shared_inventory=runner_uses_single_imagefolder,
                                               class_folders=ext.classes, sample_count=len(ext)),
        "plantvillage_checkpoint_preprocessing": status_record(
            "VERIFIED" if pv_transform_ok else "ERROR",
            "PlantVillage verification runner applies the preprocessing flag stored for each checkpoint.",
            evaluator_uses_spec_transform=pv_transform_ok),
    }

    # Check checkpoint associations, strict loading, digests, parameter totals,
    # and repeated batch-one timings (measurement variability is recorded).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_hash = verify_checkpoint()
    integrity_path = ROOT / "configs/model_integrity.json"
    integrity_all = json.loads(integrity_path.read_text(encoding="utf-8"))
    resource_saved = {}
    final_path = ROOT / "results/final_results.json"
    if final_path.is_file():
        resource_saved = json.loads(final_path.read_text(encoding="utf-8")).get("resource_measurements", {})
    for key, spec in MODEL_SPECS.items():
        checkpoint = spec["checkpoint"]
        model = build_model(key, device=device, load_checkpoint=True)
        params = sum(p.numel() for p in model.parameters())
        latency = measure_latency(model, device)
        saved_latency = resource_saved.get(key, {}).get("inference_latency_ms_batch1")
        latency_median = statistics.median(latency)
        latency_cv = (statistics.pstdev(latency) / statistics.mean(latency)) if len(latency)>1 else 0.0
        if device.type == "cuda":
            torch.cuda.empty_cache()
        saved_trials = resource_saved.get(key, {}).get("latency_trials_ms_batch1", [])
        saved_trial_median_ok = (len(saved_trials) == 3 and saved_latency is not None
                                 and abs(statistics.median(saved_trials)-saved_latency) < 1e-8)
        saved_trial_cv = (statistics.pstdev(saved_trials)/statistics.mean(saved_trials)) if len(saved_trials)>1 else None
        audit["models"][key] = {
            "checkpoint_exists_and_strictly_loads": status_record("VERIFIED", "build_model(strict state-dict load) completed.",
                                                                   path=str(checkpoint.relative_to(ROOT))),
            "checkpoint_association": {
                "expected": str(checkpoint.relative_to(ROOT)),
                "plantvillage_result": (json.loads((ROOT/"results/audit"/f"{key}_plantvillage_verified.json").read_text(encoding="utf-8")).get("checkpoint")
                                         if (ROOT/"results/audit"/f"{key}_plantvillage_verified.json").exists() else None),
                "plantdoc_result": (json.loads((ROOT/"results/domain_shift"/f"{key}_plantdoc_test_results.json").read_text(encoding="utf-8")).get("checkpoint")
                                    if (ROOT/"results/domain_shift"/f"{key}_plantdoc_test_results.json").exists() else None),
            },
            "sha256": digest(checkpoint),
            "file_size_bytes": checkpoint.stat().st_size,
            "parameter_count": params,
            "preprocessing_clahe": spec["clahe"],
            "repeated_latency_ms_per_image": [round(x, 4) for x in latency],
            "latency_median_ms_per_image": round(latency_median, 4),
            "latency_trial_cv": round(latency_cv, 6),
            "saved_latency_ms_per_image": saved_latency,
            "saved_latency_trials_ms_per_image": saved_trials,
            "saved_latency_trial_cv": saved_trial_cv,
            "saved_latency_trials_median_matches_record": saved_trial_median_ok,
            "saved_latency_relative_delta": (round(abs(latency_median-saved_latency)/saved_latency, 4)
                                               if saved_latency else None),
            "saved_parameter_count_matches": (params == resource_saved.get(key, {}).get("parameter_count")),
            "saved_file_size_matches": (checkpoint.stat().st_size == resource_saved.get(key, {}).get("checkpoint_size_bytes")),
        }
    audit["findings"]["checkpoint_and_resource_integrity"] = {
        "four_checkpoints": status_record("VERIFIED" if len(audit["models"]) == 4 else "ERROR",
                                           "All four named model specs were strictly loaded without training."),
        "hybrid_expected_sha256": status_record("VERIFIED" if expected_hash["matches"] else "ERROR",
                                                "Hybrid digest compared with configs/model_integrity.json.",
                                                verification=expected_hash),
        "checkpoint_associations": status_record(
            "VERIFIED" if all(v["checkpoint_association"]["plantvillage_result"] == v["checkpoint_association"]["expected"]
                              and v["checkpoint_association"]["plantdoc_result"] == v["checkpoint_association"]["expected"]
                              for v in audit["models"].values()) else "ERROR",
            "Saved evaluation records identify the same checkpoint path loaded from each MODEL_SPECS entry.",
            assignments={k:v["checkpoint_association"] for k,v in audit["models"].items()}),
        "parameter_count_and_file_size": status_record(
            "VERIFIED" if all(m["saved_parameter_count_matches"] and m["saved_file_size_matches"] for m in audit["models"].values()) else "ERROR",
            "Recomputed parameter counts and file byte sizes compared with final_results.json.",
            values={k: {"parameters": v["parameter_count"], "bytes": v["file_size_bytes"]} for k,v in audit["models"].items()}),
        "latency_repeatability": status_record(
            "VERIFIED" if all(m["latency_trial_cv"] < 0.10 and m["saved_latency_trials_median_matches_record"]
                              and m["saved_latency_trial_cv"] is not None and m["saved_latency_trial_cv"] < 0.10
                              for m in audit["models"].values()) else "WARNING",
            "Three 60-iteration batch-one model-forward timing trials after eight warmups; CUDA events measure GPU execution. Saved median and trial coefficient of variation are validated. CV <10% is the repeatability criterion.",
            device=str(device),
            live_trial_cv={k:v["latency_trial_cv"] for k,v in audit["models"].items()},
            saved_trial_cv={k:v["saved_latency_trial_cv"] for k,v in audit["models"].items()},
            saved_trials_match={k:v["saved_latency_trials_median_matches_record"] for k,v in audit["models"].items()},
            live_vs_saved_delta={k:v["saved_latency_relative_delta"] for k,v in audit["models"].items()}),
        "latency_cross_run_consistency": status_record(
            "WARNING" if any((m["saved_latency_relative_delta"] or 0) > 0.10 for m in audit["models"].values()) else "VERIFIED",
            "Comparison of independent audit-run medians with the saved report-run medians; >10% deviation is reported as GPU/runtime variability even when each run's internal trials are stable.",
            relative_delta={k:v["saved_latency_relative_delta"] for k,v in audit["models"].items()}),
    }

    # Strict re-evaluation artifact metrics must agree with confusion matrices,
    # sample supports, saved-vs-verified checks, domain gaps, and manuscript values.
    pv_path = ROOT / "results/audit"
    domain_path = ROOT / "results/domain_shift"
    all_metric_checks = {}
    paper_text = (ROOT / "research/paper.md").read_text(encoding="utf-8") if (ROOT / "research/paper.md").exists() else ""
    paper_reported = []
    paper_table_checks = []
    published = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
    consistency = published.get("saved_vs_verified_consistency", {})
    paired = {}
    for key in MODEL_SPECS:
        pv_file = pv_path / f"{key}_plantvillage_verified.json"
        ext_file = domain_path / f"{key}_plantdoc_test_results.json"
        pv_result = json.loads(pv_file.read_text(encoding="utf-8")) if pv_file.exists() else None
        ext_result = json.loads(ext_file.read_text(encoding="utf-8")) if ext_file.exists() else None
        per_model = {}
        for label, result, class_ids in (
            ("plantvillage", pv_result, list(range(10))),
            ("plantdoc", ext_result, list(range(10))),
        ):
            if result is None:
                per_model[label] = status_record("ERROR", "Evaluation result file is missing.")
                continue
            cm = np.asarray(result["confusion_matrix"], dtype=np.int64)
            if cm.shape != (10, 10):
                per_model[label] = status_record("ERROR", "Confusion matrix must retain all ten model outputs.", shape=list(cm.shape))
                continue
            support = int(cm.sum())
            class_ids = list(range(10)) if label == "plantvillage" else [CLASS_NAMES.index(n) for n in ext_result["mapped_classes"]]
            independent = metrics_from_cm(cm, class_ids)
            actual = result["metrics"] if label == "plantvillage" else result["metrics"]
            metric_names = {"accuracy":"accuracy", "precision_macro":"precision_macro", "recall_macro":"recall_macro", "f1_macro":"f1_macro"}
            comparisons = {}
            for metric in metric_names:
                if metric not in actual:
                    continue
                comparisons[metric] = {"from_confusion": independent[metric], "stored": actual[metric],
                                      "matches_1e_6": abs(independent[metric]-actual[metric]) <= 1e-6}
            labels = result.get("confusion_matrix_labels", CLASS_NAMES)
            class_checks = {}
            class_support_matches_inventory = {}
            for class_id in class_ids:
                class_name = labels[class_id]
                tp = int(cm[class_id, class_id])
                predicted_support = int(cm[:, class_id].sum())
                true_support = int(cm[class_id, :].sum())
                cp = tp/predicted_support if predicted_support else 0.0
                cr = tp/true_support if true_support else 0.0
                cf = 2*cp*cr/(cp+cr) if cp+cr else 0.0
                stored_class = result.get("per_class", {}).get(class_name, {})
                expected_supports = (ext_counts if label == "plantdoc" else
                                     split_inventory["test"]["per_class_support"]["counts"])
                class_support_matches_inventory[class_name] = true_support == expected_supports.get(class_name)
                class_checks[class_name] = all([
                    abs(cp-float(stored_class.get("precision", -100))) <= 1e-6,
                    abs(cr-float(stored_class.get("recall", -100))) <= 1e-6,
                    abs(cf-float(stored_class.get("f1-score", -100))) <= 1e-6,
                    true_support == int(stored_class.get("support", -1)),
                ])
            expected_n = 2054 if label == "plantvillage" else 83
            labels_ok = labels == CLASS_NAMES
            ok = (support == expected_n and labels_ok and all(x["matches_1e_6"] for x in comparisons.values())
                  and all(class_checks.values()) and all(class_support_matches_inventory.values()))
            per_model[label] = status_record("VERIFIED" if ok else "ERROR",
                                             "Confusion-matrix aggregate and per-class metric reconstruction, label order, and support check.",
                                             sample_count=support, confusion_labels_match=labels_ok,
                                             metric_comparison=comparisons, per_class_checks=class_checks,
                                             class_support_matches_dataset=class_support_matches_inventory)
        saved_ok = all(v.get("matches_within_1e-6", False)
                       for v in consistency.get(key, {}).values()) if consistency.get(key) else False
        per_model["legacy_plantvillage_metric_reproduction"] = status_record(
            "VERIFIED" if saved_ok else "WARNING",
            "Saved-versus-reverified metrics from final_results.json; absence or mismatch is surfaced.",
            metrics=consistency.get(key, {}))
        all_metric_checks[key] = per_model
        if pv_result and ext_result:
            paired_pv = metrics_from_cm(pv_result["confusion_matrix"], MAPPED_IDS)
            ext_ids = [CLASS_NAMES.index(n) for n in ext_result["mapped_classes"]]
            ext_metrics = metrics_from_cm(ext_result["confusion_matrix"], ext_ids)
            paired[key] = {
                "plantvillage_mapped": paired_pv,
                "plantdoc_mapped": ext_metrics,
                "accuracy_gap": paired_pv["accuracy"]-ext_metrics["accuracy"],
                "macro_f1_gap": paired_pv["f1_macro"]-ext_metrics["f1_macro"],
            }
            comparison = published.get("model_comparison", [])
            row = next((x for x in comparison if x.get("model_key")==key), None)
            if row:
                paper_reported.append({
                    "model": key,
                    "paper_contains_plantdoc_accuracy": f"{ext_metrics['accuracy']*100:.2f}%" in paper_text,
                    "paper_contains_plantdoc_macro_f1": f"{ext_metrics['f1_macro']*100:.2f}%" in paper_text,
                    "paper_contains_plantvillage_accuracy": f"{pv_result['metrics']['accuracy']*100:.2f}%" in paper_text,
                    "paper_contains_plantvillage_macro_f1": f"{pv_result['metrics']['f1_macro']*100:.2f}%" in paper_text,
                    "paper_contains_hybrid_accuracy_gap": True if key!="hybrid" else f"{paired[key]['accuracy_gap']*100:.2f}" in paper_text,
                    "paper_contains_hybrid_f1_gap": True if key!="hybrid" else f"{paired[key]['macro_f1_gap']*100:.2f}" in paper_text,
                })
            display_name = {
                "mobilenet": "MobileNetV3-Small",
                "mobilenet_clahe": "MobileNetV3-Small + CLAHE",
                "deit_tiny": "DeiT-Tiny",
                "hybrid": "MobileNetV3-Small + DeiT-Tiny",
            }[key]
            table_line = next((line for line in paper_text.splitlines()
                               if line.startswith("|") and len(line.split("|")) > 2
                               and line.split("|")[1].strip() == display_name), None)
            cells = [cell.strip().strip("*") for cell in table_line.split("|")] if table_line else []
            resource = published.get("resource_measurements", {}).get(key, {})
            expected_cells = {
                "pv_accuracy": (2, f"{pv_result['metrics']['accuracy']*100:.2f}%"),
                "pv_macro_f1": (3, f"{pv_result['metrics']['f1_macro']*100:.2f}%"),
                "external_accuracy": (4, f"{ext_metrics['accuracy']*100:.2f}%"),
                "external_macro_f1": (5, f"{ext_metrics['f1_macro']*100:.2f}%"),
                "accuracy_gap_pp": (6, f"{paired[key]['accuracy_gap']*100:.2f} pp"),
                "macro_f1_gap_pp": (7, f"{paired[key]['macro_f1_gap']*100:.2f} pp"),
                "parameter_count": (8, f"{resource.get('parameter_count', -1):,}"),
                "checkpoint_mib": (9, f"{resource.get('checkpoint_size_mib', -1):.2f}"),
                "latency_ms": (10, f"{resource.get('inference_latency_ms_batch1', -1):.2f}"),
            }
            paper_table_checks.append({
                "model": key,
                "row_found": table_line is not None,
                "checks": {name: bool(table_line and len(cells) > index and cells[index] == expected)
                           for name, (index, expected) in expected_cells.items()},
            })
    audit["independent_metric_recalculation"] = all_metric_checks
    audit["paired_domain_gaps"] = paired
    audit["saved_paper_value_check"] = paper_reported
    audit["paper_table_check"] = paper_table_checks
    paper_ok = (bool(paper_reported) and all(all(v for name,v in row.items() if name != "model") for row in paper_reported)
                and bool(paper_table_checks)
                and all(row["row_found"] and all(row["checks"].values()) for row in paper_table_checks))
    audit["findings"]["paper_numeric_values"] = status_record(
        "VERIFIED" if paper_ok and clahe_transform_ok else "ERROR",
        "Accuracy, macro-F1, domain gaps, parameter counts, checkpoint sizes, and displayed latency cells are checked against current artifacts; results are valid only when checkpoint-specific preprocessing also passes.",
        text_checks=paper_reported, table_checks=paper_table_checks,
    )

    # Check existing bootstrap artifact by regenerating its intervals deterministically.
    boot_path = audit_dir / "stratified_bootstrap_intervals.json"
    bootstrap_ok = False
    if boot_path.is_file() and all((pv_path/f"{k}_plantvillage_verified.json").exists() and (domain_path/f"{k}_plantdoc_test_results.json").exists() for k in MODEL_SPECS):
        stored_boot = json.loads(boot_path.read_text(encoding="utf-8"))
        rng = np.random.default_rng(stored_boot.get("seed", 20260924))
        recomputed = {}
        for key in MODEL_SPECS:
            pvr=json.loads((pv_path/f"{key}_plantvillage_verified.json").read_text(encoding="utf-8"))
            exr=json.loads((domain_path/f"{key}_plantdoc_test_results.json").read_text(encoding="utf-8"))
            mapped=[CLASS_NAMES.index(name) for name in exr["mapped_classes"]]
            recomputed[key]={
                "plantvillage_full_10_classes":bootstrap_reference(pvr["confusion_matrix"],list(range(10)),rng),
                "plantvillage_mapped_9_classes":bootstrap_reference(pvr["confusion_matrix"],mapped,rng),
                "plantdoc_mapped_9_classes":bootstrap_reference(exr["confusion_matrix"],mapped,rng),
            }
        bootstrap_ok = recomputed == {k:{s:{m:v for m,v in metrics.items() if m.endswith("ci")} for s,metrics in per.items()} for k,per in stored_boot.get("models",{}).items()}
    audit["findings"]["bootstrap"] = status_record("VERIFIED" if bootstrap_ok else "ERROR",
                                                    "10,000 stratified multinomial resamples regenerated from saved confusion matrices and compared with recorded percentile intervals.",
                                                    seed=20260924, replicates=10_000)

    # Validate current stored Grad-CAM++ outputs and directly smoke-test a fresh map.
    explain_files = list((ROOT/"results/explainability").glob("plant*/*gradcampp*.png"))
    shape_checks=[]
    for path in explain_files:
        try:
            with Image.open(path) as im:
                shape_checks.append(im.size == (224,224) and path.stat().st_size > 0)
        except Exception:
            shape_checks.append(False)
    test_sample = pv_test.samples[0][0]
    with Image.open(test_sample) as im:
        sample = im.convert("RGB")
        x = make_transform()(sample)
        hybrid = build_model("hybrid", device=device)
        explanation = explain_pil(hybrid, sample, model_key="hybrid")
        with torch.inference_mode():
            probs = hybrid(x.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()
    branch_ok = set(explanation["branch_cams"]) == {"mobilenet", "deit"}
    cams = list(explanation["branch_cams"].values()) + [explanation["combined_cam"]]
    cam_ok = all(c.shape == (224,224) and np.isfinite(c).all() and float(np.ptp(c)) > 0.1 for c in cams)
    score_diff = abs(float(probs.max())-explanation["confidence"])
    audit["findings"]["gradcampp"] = {
        "stored_outputs": status_record("VERIFIED" if explain_files and all(shape_checks) else "ERROR",
                                        "Existing Grad-CAM++ PNG dimensions and non-empty files checked.",
                                        files_checked=len(explain_files), all_224x224=all(shape_checks)),
        "fresh_smoke": status_record("VERIFIED" if branch_ok and cam_ok and score_diff < 1e-6 else "ERROR",
                                     "Fresh hybrid attribution forward pass; verifies both branch maps, finite/nonconstant maps, and prediction confidence parity.",
                                     branches=sorted(explanation["branch_cams"]), branch_maps_valid=cam_ok,
                                     confidence_absolute_difference=score_diff,
                                     predicted_class_index=explanation["predicted_index"]),
    }

    # Individual project-specific status for confusion matrices and closed-set errors.
    cm_ok = all(row[label]["status"] == "VERIFIED" for row in all_metric_checks.values() for label in ("plantvillage","plantdoc"))
    audit["findings"]["confusion_matrices_and_metrics"] = status_record(
        "VERIFIED" if cm_ok else "ERROR",
        "PV and PlantDoc supports, full 10-output confusion matrix dimensions, and accuracy/macro P/R/F1 reconstructed independently from matrix counts.")
    audit["findings"]["unsupported_predictions"] = status_record(
        "VERIFIED" if cm_ok else "ERROR",
        "Full ten-column external confusion matrices retain unsupported model outputs; they are not discarded from accuracy or mapped-class precision.")

    # Summarize historical and current paper consistency. A pre-correction audit
    # preserves the known preprocessing defect rather than hiding it.
    prior = audit_dir / "pre_correction_domain_shift/finding.json"
    if prior.is_file():
        audit["historical_preprocessing_finding"] = json.loads(prior.read_text(encoding="utf-8"))
    historical_corrected = clahe_transform_ok
    if args.initial:
        audit["run_stage"] = "before CLAHE preprocessing correction"
        json_path = audit_dir / "phase1_initial_audit.json"
    else:
        audit["run_stage"] = "current implementation"
        json_path = audit_dir / "reproducibility_audit.json"
    json_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")

    lines = ["# Phase 1 Reproducibility Audit", "", f"**Stage:** {audit['run_stage']}",
             "", "No training was performed.", "", "## Overall checks", ""]
    status_items=[]
    def collect(value, path=""):
        if isinstance(value,dict):
            if value.get("status") in {"VERIFIED","WARNING","ERROR","NOT TESTED"}:
                status_items.append((path,value))
            else:
                for k,v in value.items(): collect(v, f"{path}/{k}" if path else k)
    collect(audit.get("findings",{}))
    for key,value in status_items:
        lines.append(f"- **{value['status']}** `{key}` — {value['detail']}")
    lines += ["", "## Checkpoint and model resource audit", "",
              "| Model | SHA-256 | Bytes | Parameters | Repeated latency trials (ms/image) | Saved latency delta |", "|---|---|---:|---:|---|---:|"]
    for key,v in audit["models"].items():
        lines.append(f"| {key} | `{v['sha256']}` | {v['file_size_bytes']} | {v['parameter_count']:,} | {', '.join(f'{x:.3f}' for x in v['repeated_latency_ms_per_image'])} | {v['saved_latency_relative_delta']} |")
    lines += ["", "## Independently reconstructed metrics", "",
              "Metrics below are regenerated from the saved confusion matrices. Accuracy uses all samples; macro P/R/F1 are averaged over ten PlantVillage classes or nine mapped PlantDoc true classes, respectively.", "",
              "| Model | PV accuracy / macro F1 | PlantDoc n | PlantDoc accuracy / macro F1 | Same-class accuracy / F1 gap |", "|---|---:|---:|---:|---:|"]
    for key,p in paired.items():
        row=all_metric_checks[key]
        pv=json.loads((pv_path/f"{key}_plantvillage_verified.json").read_text(encoding="utf-8"))["metrics"]
        ex=json.loads((domain_path/f"{key}_plantdoc_test_results.json").read_text(encoding="utf-8"))["metrics"]
        lines.append(f"| {key} | {pv['accuracy']:.4%} / {pv['f1_macro']:.4%} | {ex['total']} | {ex['accuracy']:.4%} / {ex['f1_macro']:.4%} | {p['accuracy_gap']:.4%} / {p['macro_f1_gap']:.4%} |")
    lines += ["", "## Discrepancies and limits", ""]
    if prior.is_file():
        old=json.loads(prior.read_text(encoding="utf-8"))
        lines.append(f"- **ERROR FOUND THEN; CORRECTION STATUS: {'VERIFIED' if historical_corrected else 'ERROR'}:** pre-correction PlantDoc evaluation skipped CLAHE for `mobilenet_clahe`. Its old scores ({old['pre_correction_metrics']['accuracy']:.4%} accuracy, {old['pre_correction_metrics']['f1_macro']:.4%} macro F1) are preserved in `results/audit/pre_correction_domain_shift/` and superseded. The manuscript and active result files must use only the corrected rerun.")
    lines += ["- Image-folder class consistency and manifests were verified mechanically; original PlantVillage annotations and PlantDoc label semantics were not independently adjudicated.",
              "- Cross-dataset duplication was checked by exact SHA-256 only. Perceptual near-duplicates were not tested.",
              "- Latency was measured in three short repeated trials on the recorded device. This does not establish hardware-independent or long-run timing repeatability.",
              "- Saved-run and independent audit-run latency medians are compared separately; >10% drift indicates runtime/device variability and is not hidden by within-run trial stability.",
              "- Grad-CAM++ maps were checked for finite, nonconstant branch outputs and confidence parity; this validates execution, not explanation faithfulness or localization.",
              "- The PlantDoc subset remains only 83 images. Bootstrap intervals condition on these samples and do not resolve dataset selection or mapping uncertainty.",
              "", f"Machine-readable details: `{json_path.relative_to(ROOT).as_posix()}`.", ""]
    report = ROOT / "research/audit_report.md"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"stage":audit["run_stage"],"json":str(json_path.relative_to(ROOT)),"report":str(report.relative_to(ROOT)),
                      "clahe_preprocessing_verified":clahe_transform_ok,"metrics_verified":cm_ok,
                      "bootstrap_verified":bootstrap_ok,"device":str(device)},indent=2))


if __name__ == "__main__":
    main()
