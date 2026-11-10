"""Check all newly added PlantSeg, uncertainty, and paired-test values in paper.md."""
import json
from pathlib import Path

from src.inference.common import MODEL_SPECS, ROOT


def model_row(lines, header_prefix, model_name):
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    except StopIteration:
        return None
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip().strip("*") for cell in line.split("|")]
        if len(cells) > 1 and cells[1] == model_name:
            return cells
    return None


def percent_ci(values):
    return f"[{values[0]*100:.2f}, {values[1]*100:.2f}]%"


def main():
    paper = (ROOT / "research/paper.md").read_text(encoding="utf-8")
    lines = paper.splitlines()
    result = json.loads((ROOT / "results/final_research_results.json").read_text(encoding="utf-8"))
    plantseg_audit = json.loads((ROOT / "results/audit/plantseg_v7_evaluation_audit.json").read_text(encoding="utf-8"))
    checks = {}
    failures = []
    for key, spec in MODEL_SPECS.items():
        display_name = "MobileNetV3-Small + DeiT-Tiny" if key == "hybrid" else spec["name"]
        record = result["models"][key]
        external = record["external"]["plantseg_v7"]["metrics"]
        gap = record["domain_gap"]["plantseg_v7"]
        table = model_row(lines, "| Model | PlantSeg accuracy", display_name)
        checks[key] = {
            "plantseg_row_found": table is not None,
            "plantseg_accuracy_matches": bool(table and table[2] == f"{external['accuracy']*100:.2f}%"),
            "plantseg_macro_f1_matches": bool(table and table[3] == f"{external['f1_macro']*100:.2f}%"),
            "plantseg_accuracy_gap_matches": bool(table and table[4] == f"{gap['accuracy']*100:.2f} pp"),
            "plantseg_macro_f1_gap_matches": bool(table and table[5] == f"{gap['macro_f1']*100:.2f} pp"),
        }
        ci_row = model_row(lines, "| Model | PlantDoc accuracy 95% CI", display_name)
        expected_doc = record["external"]["plantdoc"]["uncertainty"]
        expected_seg = record["external"]["plantseg_v7"]["uncertainty"]
        checks[key].update({
            "confidence_interval_row_found": ci_row is not None,
            "plantdoc_accuracy_ci_matches": bool(ci_row and ci_row[2] == percent_ci(expected_doc["accuracy_95_ci"])),
            "plantdoc_macro_f1_ci_matches": bool(ci_row and ci_row[3] == percent_ci(expected_doc["macro_f1_95_ci"])),
            "plantseg_accuracy_ci_matches": bool(ci_row and ci_row[4] == percent_ci(expected_seg["accuracy_95_ci"])),
            "plantseg_macro_f1_ci_matches": bool(ci_row and ci_row[5] == percent_ci(expected_seg["macro_f1_95_ci"])),
        })
        failures.extend(f"{key}:{name}" for name, okay in checks[key].items() if not okay)

    paired_checks = {}
    for slug, artifact in result["paired_comparisons"].items():
        comparison = next(
            item for item in artifact["comparisons"]
            if {item["model_a"], item["model_b"]} == {"deit_tiny", "hybrid"}
        )
        sign = 1 if comparison["model_a"] == "deit_tiny" else -1
        acc = comparison["accuracy_difference_a_minus_b"] * sign * 100
        f1 = comparison["macro_f1_difference_a_minus_b"] * sign * 100
        acc_ci = [value * sign * 100 for value in comparison["accuracy_difference_paired_stratified_bootstrap_95_ci"]]
        f1_ci = [value * sign * 100 for value in comparison["macro_f1_difference_paired_stratified_bootstrap_95_ci"]]
        if sign < 0:
            acc_ci.reverse()
            f1_ci.reverse()
        accuracy_phrase = f"{acc:.2f} percentage points" if slug == "plantdoc" else f"{acc:.2f} points"
        tokens = [
            accuracy_phrase,
            f"[{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]",
            f"{f1:.2f} points",
            f"[{f1_ci[0]:.2f}, {f1_ci[1]:.2f}]",
        ]
        paired_checks[slug] = {token: token in paper for token in tokens}
        failures.extend(f"{slug}:paired:{token}" for token, okay in paired_checks[slug].items() if not okay)

    audit_ok = plantseg_audit.get("audit_status") == "VERIFIED"
    report = {
        "audit_status": "VERIFIED" if not failures and audit_ok else "ERROR",
        "source_result_artifact": "results/final_research_results.json",
        "plantseg_prediction_audit_status": plantseg_audit.get("audit_status"),
        "model_table_checks": checks,
        "paired_deit_hybrid_checks": paired_checks,
        "failed_checks": failures,
        "note": "This checks paper values and strings against saved evaluation artifacts; it does not review scientific interpretation, source labels, license suitability, or manuscript venue fit.",
    }
    out = ROOT / "results/audit/paper_value_verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['audit_status']}: wrote {out.relative_to(ROOT)}")
    if report["audit_status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
