"""SHA-256 checksum verification for the frozen hybrid checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

from src.inference.common import MODEL_SPECS, ROOT

CHECKPOINT = MODEL_SPECS["hybrid"]["checkpoint"]
EXPECTED_FILE = ROOT / "configs/model_integrity.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path=CHECKPOINT, expected_file=EXPECTED_FILE):
    if not Path(path).is_file():
        return {"status": "missing", "path": str(path), "actual_sha256": None,
                "expected_sha256": None, "matches": False}
    actual = sha256_file(Path(path))
    if not Path(expected_file).is_file():
        return {"status": "expected_hash_missing", "path": str(path),
                "actual_sha256": actual, "expected_sha256": None, "matches": False}
    expected_data = json.loads(Path(expected_file).read_text(encoding="utf-8"))
    expected = expected_data.get("sha256", "").lower()
    matches = bool(expected) and actual.lower() == expected
    return {"status": "verified" if matches else "mismatch", "path": str(path),
            "actual_sha256": actual, "expected_sha256": expected or None,
            "matches": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-expected", action="store_true",
                        help="Write the current hybrid checkpoint hash as the expected value")
    args = parser.parse_args()
    if args.set_expected:
        value = sha256_file(CHECKPOINT)
        EXPECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED_FILE.write_text(json.dumps({
            "algorithm": "SHA-256",
            "checkpoint": "models/hybrid_best.pth",
            "sha256": value,
            "purpose": "Detect accidental or unauthorized checkpoint changes; not a signature.",
        }, indent=2) + "\n", encoding="utf-8")
    result = verify_checkpoint()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["matches"] else 1)


if __name__ == "__main__":
    main()
