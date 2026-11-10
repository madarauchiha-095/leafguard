"""Local LeafGuard web application with session auth and model inference APIs."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.explainability.gradcampp import explain_pil, overlay_cam
from src.inference.common import CLASS_NAMES, MODEL_SPECS, ROOT, build_model, make_transform
from src.security.integrity import sha256_file

WEB_DIR = Path(__file__).resolve().parent / "web"
STATE_DIR = Path(__file__).resolve().parent / "state"
DATABASE = STATE_DIR / "users.sqlite3"
SESSION_KEY_FILE = STATE_DIR / "session.key"
MANIFEST = ROOT / "research/reproducibility_manifest.json"
RESULTS = ROOT / "results/final_research_results.json"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_FILES_PER_REQUEST = 8  # Per request only; the browser streams folders in chunks.
MAX_REQUEST_BYTES = MAX_FILE_BYTES * MAX_FILES_PER_REQUEST
MODEL_KEYS = tuple(MODEL_SPECS)
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_ATTEMPTS = 6
ANALYSIS_WINDOW_SECONDS = 60 * 60
ANALYSIS_IMAGE_LIMIT = 120
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "testserver")


def _session_secret() -> str:
    configured = os.environ.get("LEAFGUARD_SESSION_SECRET")
    if configured:
        return configured
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_KEY_FILE.exists():
        value = SESSION_KEY_FILE.read_text(encoding="ascii").strip()
        if len(value) >= 48:
            return value
    value = secrets.token_urlsafe(48)
    SESSION_KEY_FILE.write_text(value, encoding="ascii")
    return value


@contextmanager
def _db():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_database() -> None:
    with _db() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                images_analyzed INTEGER NOT NULL DEFAULT 0,
                agreement_count INTEGER NOT NULL DEFAULT 0,
                confidence_sum REAL NOT NULL DEFAULT 0,
                prediction_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
                client_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                device TEXT NOT NULL,
                models_json TEXT NOT NULL,
                UNIQUE(run_id, client_id)
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS analysis_runs_user_created ON analysis_runs(user_id, created_at DESC)")


class ModelService:
    def __init__(self):
        self._lock = threading.RLock()
        self._models: dict | None = None
        self.device: torch.device | None = None
        self._integrity: dict[str, dict] | None = None

    def integrity(self) -> dict[str, dict]:
        if self._integrity is not None:
            return self._integrity
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            expected = manifest.get("checkpoints", {})
        except (OSError, ValueError, TypeError):
            expected = {}
        result = {}
        for key, spec in MODEL_SPECS.items():
            path = spec["checkpoint"]
            actual = sha256_file(path) if path.is_file() else None
            expected_hash = expected.get(key, {}).get("sha256")
            result[key] = {
                "model": spec["name"], "path": str(path.relative_to(ROOT)),
                "sha256": actual, "expected_sha256": expected_hash,
                "verified": bool(actual and expected_hash and hmac.compare_digest(actual, expected_hash)),
            }
        self._integrity = result
        return result

    def load(self) -> tuple[dict, torch.device]:
        with self._lock:
            if self._models is not None and self.device is not None:
                return self._models, self.device
            integrity = self.integrity()
            invalid = [item["model"] for item in integrity.values() if not item["verified"]]
            if invalid:
                raise RuntimeError("Checkpoint integrity verification failed: " + ", ".join(invalid))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded = {key: build_model(key, device=device) for key in MODEL_KEYS}
            self._models, self.device = loaded, device
            return loaded, device

    def predict(self, image: Image.Image) -> tuple[dict, str]:
        with self._lock:
            models, device = self.load()
            predictions = {}
            for key, model in models.items():
                tensor = make_transform(clahe=MODEL_SPECS[key]["clahe"])(image).unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    probabilities = model(tensor).softmax(dim=1)[0].detach().cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000
                order = probabilities.argsort()[::-1][:3]
                predictions[key] = {
                    "model": MODEL_SPECS[key]["name"],
                    "predicted_index": int(order[0]),
                    "predicted_class": CLASS_NAMES[int(order[0])],
                    "confidence": float(probabilities[order[0]]),
                    "latency_ms": elapsed,
                    "top3": [{"class": CLASS_NAMES[int(i)], "confidence": float(probabilities[i])} for i in order],
                }
            return predictions, str(device)

    def explain(self, key: str, image: Image.Image) -> dict:
        if key not in MODEL_SPECS:
            raise ValueError("Unknown model selection")
        with self._lock:
            models, _ = self.load()
            result = explain_pil(models[key], image, model_key=key, clahe=MODEL_SPECS[key]["clahe"])
            branches = {name: _png_data(overlay_cam(result["original"], cam)) for name, cam in result["branch_cams"].items()}
            return {
                "model": MODEL_SPECS[key]["name"], "class_name": result["class_name"],
                "confidence": result["confidence"], "top3": result["top3"],
                "original": _png_data(result["original"]), "overlay": _png_data(result["overlay"]),
                "branches": branches,
                "interpretation": "Grad-CAM++ is qualitative attribution, not lesion localization or calibrated confidence.",
            }


def _png_data(array) -> str:
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


MODELS = ModelService()
LOGIN_EVENTS: dict[str, deque] = defaultdict(deque)
LOGIN_LOCK = threading.Lock()


def _check_rate_limit(request: Request, bucket: str) -> bool:
    host = _client_address(request)
    key = f"{bucket}:{host}"
    now = time.monotonic()
    with LOGIN_LOCK:
        events = LOGIN_EVENTS[key]
        while events and now - events[0] > LOGIN_WINDOW_SECONDS:
            events.popleft()
        if len(events) >= LOGIN_ATTEMPTS:
            return False
        events.append(now)
    return True


def _clear_rate_limit(request: Request, bucket: str) -> None:
    host = _client_address(request)
    with LOGIN_LOCK:
        LOGIN_EVENTS.pop(f"{bucket}:{host}", None)


def _client_address(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if peer in {"127.0.0.1", "::1"}:
        forwarded = request.headers.get("cf-connecting-ip", "").strip()
        try:
            return ipaddress.ip_address(forwarded).compressed
        except ValueError:
            pass
    return peer


def _allowed_hosts() -> list[str]:
    configured = os.environ.get("LEAFGUARD_ALLOWED_HOSTS", "")
    extra = [host.strip() for host in configured.split(",") if host.strip()]
    if any(host == "*" for host in extra):
        raise ValueError("LEAFGUARD_ALLOWED_HOSTS must not contain a wildcard host.")
    return list(dict.fromkeys((*DEFAULT_ALLOWED_HOSTS, *extra)))


def _csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


def _mutation_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    host = request.headers.get("host", "").lower()
    if origin:
        from urllib.parse import urlparse
        origin_host = urlparse(origin).netloc.lower()
        if origin_host != host:
            return False
    supplied = request.headers.get("x-csrf-token", "")
    expected = request.session.get("csrf", "")
    return bool(supplied and expected and hmac.compare_digest(supplied, expected))


ANALYSIS_EVENTS: dict[int, deque[tuple[float, int]]] = defaultdict(deque)


def _reserve_analysis_quota(user_id: int, file_count: int) -> int:
    now = time.monotonic()
    with LOGIN_LOCK:
        events = ANALYSIS_EVENTS[user_id]
        while events and now - events[0][0] >= ANALYSIS_WINDOW_SECONDS:
            events.popleft()
        used = sum(count for _, count in events)
        if used + file_count > ANALYSIS_IMAGE_LIMIT:
            return max(1, int(ANALYSIS_WINDOW_SECONDS - (now - events[0][0])))
        events.append((now, file_count))
        return 0


def _page(name: str, auth: bool = False):
    async def endpoint(request: Request):
        if auth and not request.session.get("user_id"):
            return RedirectResponse("/login", status_code=303)
        if name in {"login.html", "register.html"} and request.session.get("user_id"):
            return RedirectResponse("/dashboard", status_code=303)
        return FileResponse(WEB_DIR / name, media_type="text/html; charset=utf-8")
    return endpoint


async def csrf(request: Request):
    response = JSONResponse({"csrf_token": _csrf_token(request)})
    response.headers["Cache-Control"] = "no-store"
    return response


async def register(request: Request):
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    if not _check_rate_limit(request, "register"):
        return JSONResponse({"error": "Too many account attempts. Try again later."}, status_code=429)
    try:
        payload = await request.json()
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not 3 <= len(username) <= 40 or not all(ch.isalnum() or ch in "._- " for ch in username):
            return JSONResponse({"error": "Use a 3-40 character username with letters, numbers, spaces, dots, dashes, or underscores."}, status_code=400)
        if not 12 <= len(password) <= 128:
            return JSONResponse({"error": "Password must be between 12 and 128 characters."}, status_code=400)
        salt = secrets.token_bytes(16)
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        with _db() as connection:
            connection.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", (username, password_hash, salt))
        return JSONResponse({"message": "Account created. Sign in to continue."}, status_code=201)
    except sqlite3.IntegrityError:
        return JSONResponse({"error": "Unable to create this account. Choose a different username."}, status_code=409)
    except (ValueError, TypeError, UnicodeError):
        return JSONResponse({"error": "Invalid account details."}, status_code=400)


async def login(request: Request):
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    if not _check_rate_limit(request, "login"):
        return JSONResponse({"error": "Too many sign-in attempts. Try again later."}, status_code=429)
    try:
        payload = await request.json()
        username, password = str(payload.get("username", "")).strip(), str(payload.get("password", ""))
        with _db() as connection:
            row = connection.execute("SELECT id, username, password_hash, salt FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        if not row:
            return JSONResponse({"error": "Username or password is incorrect."}, status_code=401)
        attempt = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), row["salt"], 310_000)
        if not hmac.compare_digest(attempt, row["password_hash"]):
            return JSONResponse({"error": "Username or password is incorrect."}, status_code=401)
        request.session.clear()
        request.session["user_id"] = row["id"]
        request.session["username"] = row["username"]
        _csrf_token(request)
        _clear_rate_limit(request, "login")
        return JSONResponse({"username": row["username"]})
    except (ValueError, TypeError, UnicodeError):
        return JSONResponse({"error": "Invalid sign-in details."}, status_code=400)


async def me(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "username": request.session.get("username"), "csrf_token": _csrf_token(request)})


async def logout(request: Request):
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    request.session.clear()
    return JSONResponse({"ok": True})


async def create_analysis_run(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    run_id = str(uuid.uuid4())
    with _db() as connection:
        connection.execute("INSERT INTO analysis_runs (id, user_id) VALUES (?, ?)", (run_id, user_id))
    return JSONResponse({"id": run_id}, status_code=201)


async def analysis_history(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    with _db() as connection:
        summary = connection.execute("""
            SELECT COUNT(*) AS total_runs,
                   COALESCE(SUM(images_analyzed), 0) AS images_analyzed,
                   COALESCE(SUM(agreement_count), 0) AS agreement_count,
                   COALESCE(SUM(confidence_sum), 0) AS confidence_sum,
                   COALESCE(SUM(prediction_count), 0) AS prediction_count
            FROM analysis_runs WHERE user_id = ?
        """, (user_id,)).fetchone()
        runs = connection.execute("""
            SELECT id, created_at, images_analyzed, agreement_count,
                   confidence_sum, prediction_count
            FROM analysis_runs WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 50
        """, (user_id,)).fetchall()
    return JSONResponse({
        "summary": {
            "total_runs": summary["total_runs"],
            "images_analyzed": summary["images_analyzed"],
            "agreement_count": summary["agreement_count"],
            "confidence_sum": summary["confidence_sum"],
            "prediction_count": summary["prediction_count"],
        },
        "runs": [dict(row) for row in runs],
    })


async def analysis_run_detail(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    run_id = request.path_params["run_id"]
    try:
        page = max(0, int(request.query_params.get("page", "0")))
    except ValueError:
        return JSONResponse({"error": "Invalid history page."}, status_code=400)
    page_size = 100
    with _db() as connection:
        run = connection.execute("""
            SELECT id, created_at, images_analyzed, agreement_count,
                   confidence_sum, prediction_count
            FROM analysis_runs WHERE id = ? AND user_id = ?
        """, (run_id, user_id)).fetchone()
        if run is None:
            return JSONResponse({"error": "Analysis run not found."}, status_code=404)
        total = connection.execute(
            "SELECT COUNT(*) FROM analysis_results WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        results = connection.execute("""
            SELECT client_id, filename, sha256, width, height, device, models_json
            FROM analysis_results WHERE run_id = ? ORDER BY id LIMIT ? OFFSET ?
        """, (run_id, page_size, page * page_size)).fetchall()
    images = []
    for row in results:
        image = dict(row)
        image["models"] = json.loads(image.pop("models_json"))
        images.append(image)
    return JSONResponse({"run": dict(run), "results": images, "total": total, "page": page, "page_size": page_size})


async def delete_analysis_run(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    with _db() as connection:
        deleted = connection.execute(
            "DELETE FROM analysis_runs WHERE id = ? AND user_id = ?",
            (request.path_params["run_id"], user_id),
        ).rowcount
    if not deleted:
        return JSONResponse({"error": "Analysis run not found."}, status_code=404)
    return JSONResponse({"ok": True})


def _checked_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("File is empty or exceeds the 10 MB per-image limit.")
    try:
        with Image.open(BytesIO(raw)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ValueError("Use JPEG, PNG, WebP, or BMP images.")
            if probe.width < 32 or probe.height < 32:
                raise ValueError("Image must be at least 32 x 32 pixels.")
            if probe.width * probe.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 20 megapixel limit.")
            probe.verify()
        with Image.open(BytesIO(raw)) as decoded:
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image could not be decoded safely.") from exc


async def analyze(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    form = await request.form(max_files=MAX_FILES_PER_REQUEST, max_fields=MAX_FILES_PER_REQUEST * 2, max_part_size=MAX_FILE_BYTES)
    files = form.getlist("files")
    client_ids = form.getlist("client_ids")
    run_id = str(form.get("run_id", ""))
    if not files or len(files) > MAX_FILES_PER_REQUEST:
        return JSONResponse({"error": f"Submit 1-{MAX_FILES_PER_REQUEST} files per processing request. Folder size is not limited by this chunk size."}, status_code=400)
    retry_after = _reserve_analysis_quota(int(request.session["user_id"]), len(files))
    if retry_after:
        return JSONResponse(
            {"error": f"Analysis limit reached. This account can process up to {ANALYSIS_IMAGE_LIMIT} images per hour."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    if run_id:
        with _db() as connection:
            exists = connection.execute(
                "SELECT 1 FROM analysis_runs WHERE id = ? AND user_id = ?",
                (run_id, request.session["user_id"]),
            ).fetchone()
        if not exists:
            return JSONResponse({"error": "Analysis run not found."}, status_code=404)
    total_bytes, rows = 0, []
    for index, upload in enumerate(files):
        raw = await upload.read(MAX_FILE_BYTES + 1)
        total_bytes += len(raw)
        file_name = PurePosixPath(str(upload.filename or "image").replace("\\", "/")).name[:160]
        client_id = str(client_ids[index])[:180] if index < len(client_ids) else f"{index}"
        digest = hashlib.sha256(raw).hexdigest()
        try:
            image = _checked_image(raw)
            predictions, device = await asyncio.to_thread(MODELS.predict, image)
            rows.append({"client_id": client_id, "filename": file_name, "sha256": digest, "width": image.width, "height": image.height, "device": device, "models": predictions})
        except ValueError as exc:
            rows.append({"client_id": client_id, "filename": file_name, "sha256": digest, "error": str(exc)})
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:
            rows.append({"client_id": client_id, "filename": file_name, "sha256": digest, "error": f"Analysis failed ({type(exc).__name__})."})
    if total_bytes > MAX_REQUEST_BYTES:
        return JSONResponse({"error": "Processing chunk exceeds its 80 MB transfer limit."}, status_code=413)
    if run_id:
        saved = [row for row in rows if "models" in row]
        if saved:
            agreements = sum(
                len({model["predicted_class"] for model in row["models"].values()}) == 1
                for row in saved
            )
            confidence_sum = sum(
                model["confidence"] for row in saved for model in row["models"].values()
            )
            prediction_count = sum(len(row["models"]) for row in saved)
            with _db() as connection:
                connection.executemany("""
                    INSERT INTO analysis_results
                    (run_id, client_id, filename, sha256, width, height, device, models_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (run_id, row["client_id"], row["filename"], row["sha256"],
                     row["width"], row["height"], row["device"],
                     json.dumps(row["models"], separators=(",", ":")))
                    for row in saved
                ])
                connection.execute("""
                    UPDATE analysis_runs
                    SET images_analyzed = images_analyzed + ?,
                        agreement_count = agreement_count + ?,
                        confidence_sum = confidence_sum + ?,
                        prediction_count = prediction_count + ?
                    WHERE id = ? AND user_id = ?
                """, (len(saved), agreements, confidence_sum, prediction_count, run_id, request.session["user_id"]))
    return JSONResponse({"results": rows})


async def explain(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    if not _mutation_allowed(request):
        return JSONResponse({"error": "Security token expired. Reload and try again."}, status_code=403)
    form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_FILE_BYTES)
    upload = form.get("file")
    model_key = str(form.get("model", ""))
    if upload is None:
        return JSONResponse({"error": "Image is required."}, status_code=400)
    retry_after = _reserve_analysis_quota(int(request.session["user_id"]), 1)
    if retry_after:
        return JSONResponse(
            {"error": f"Analysis limit reached. This account can process up to {ANALYSIS_IMAGE_LIMIT} images per hour."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    try:
        image = _checked_image(await upload.read(MAX_FILE_BYTES + 1))
        result = await asyncio.to_thread(MODELS.explain, model_key, image)
        return JSONResponse(result)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


async def research(request: Request):
    try:
        full = json.loads(RESULTS.read_text(encoding="utf-8"))
        models = {}
        for key, data in full["models"].items():
            models[key] = {
                "name": data["model"], "plantvillage": data["plantvillage_full"],
                "plantdoc": data["external"]["plantdoc"], "plantseg_v7": data["external"]["plantseg_v7"],
                "resources": {name: data.get("resources", {}).get(name) for name in ("parameter_count", "checkpoint_size_mib", "inference_latency_ms_batch1")},
            }
        return JSONResponse({"models": models, "counts": {"plantvillage": 2054, "plantdoc": 83, "plantseg_v7": 143}, "limitations": full.get("limitations", [])})
    except (OSError, ValueError, KeyError, TypeError):
        return JSONResponse({"error": "Research metrics are unavailable."}, status_code=503)


async def security_status(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    integrity = await asyncio.to_thread(MODELS.integrity)
    return JSONResponse({
        "checkpoints": integrity, "all_verified": all(row["verified"] for row in integrity.values()),
        "controls": [
            {"name": "Session cookies", "status": "Enabled", "detail": "HttpOnly, SameSite=Strict, signed session cookie."},
            {"name": "CSRF protection", "status": "Enabled", "detail": "Session-bound token required for mutations."},
            {"name": "Login throttling", "status": "Enabled", "detail": "Failed sign-in attempts are throttled by client address."},
            {"name": "Upload validation", "status": "Enabled", "detail": "Format, encoded size, and decoded pixel limits are checked."},
            {"name": "Original image retention", "status": "Not retained", "detail": "Uploaded image bytes are discarded after analysis. Prediction metadata is stored in local history until deleted."},
        ],
        "device": str(MODELS.device) if MODELS.device else ("cuda" if torch.cuda.is_available() else "cpu"),
    })


async def model_catalog(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"error": "Sign-in required."}, status_code=401)
    return JSONResponse({key: spec["name"] for key, spec in MODEL_SPECS.items()})


async def health(request: Request):
    return JSONResponse({"status": "ok", "service": "LeafGuard"})


async def artifact(request: Request):
    name = request.path_params["name"]
    allowed = {"plantvillage_vs_external_accuracy.png", "plantvillage_vs_external_macro_f1.png", "plantdoc_confusion_matrices.png", "plantseg_v7_confusion_matrices.png", "domain_gap_accuracy.png", "parameter_count_vs_latency.png"}
    if name not in allowed:
        return Response(status_code=404)
    path = ROOT / "results/plots/final" / name
    return FileResponse(path) if path.is_file() else Response(status_code=404)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if request.url.scheme == "https" or os.environ.get("LEAFGUARD_HTTPS") == "1":
            response.headers.setdefault("Strict-Transport-Security", "max-age=15552000; includeSubDomains")
        return response


@asynccontextmanager
async def lifespan(application):
    _initialize_database()
    yield


routes = [
    Route("/", _page("index.html")), Route("/login", _page("login.html")),
    Route("/register", _page("register.html")), Route("/workspace", _page("workspace.html", auth=True)),
    Route("/dashboard", _page("dashboard.html", auth=True)),
    Route("/insights", _page("insights.html")), Route("/security", _page("security.html", auth=True)),
    Route("/api/auth/csrf", csrf), Route("/api/auth/register", register, methods=["POST"]),
    Route("/api/auth/login", login, methods=["POST"]), Route("/api/auth/me", me),
    Route("/api/auth/logout", logout, methods=["POST"]),
    Route("/api/analysis/runs", create_analysis_run, methods=["POST"]),
    Route("/api/analysis/history", analysis_history),
    Route("/api/analysis/history/{run_id:str}", analysis_run_detail),
    Route("/api/analysis/history/{run_id:str}/delete", delete_analysis_run, methods=["POST"]),
    Route("/api/analyze", analyze, methods=["POST"]),
    Route("/api/explain", explain, methods=["POST"]), Route("/api/research", research),
    Route("/api/models", model_catalog), Route("/api/security", security_status), Route("/artifacts/{name:str}", artifact),
    Route("/healthz", health),
    Mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets"),
]
middleware = [
    Middleware(SessionMiddleware, secret_key=_session_secret(), session_cookie="leafguard_session", max_age=8 * 60 * 60, same_site="strict", https_only=os.environ.get("LEAFGUARD_HTTPS", "0") == "1"),
    Middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts()),
    Middleware(SecurityHeadersMiddleware),
]
app = Starlette(debug=False, routes=routes, middleware=middleware, lifespan=lifespan, max_body_size=MAX_REQUEST_BYTES + 2 * 1024 * 1024)
