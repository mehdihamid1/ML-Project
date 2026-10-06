"""Flask interface with private, bounded server-side sessions.

The in-process registry requires one Gunicorn worker. Uploads are never
executed; only registered CSV paths are passed to the tool service.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from urllib.parse import urlsplit
import os
import secrets
import shutil
import tempfile
import time
import uuid

from flask import Flask, g, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class UserSession:
    directory: Path
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    files: dict = field(default_factory=dict)
    state: dict = field(default_factory=lambda: {
        "history": [], "results": [], "activity": [], "downloads": {},
    })
    touched: float = field(default_factory=time.monotonic)
    lock: RLock = field(default_factory=RLock)
    active_requests: int = 0


class SessionStore:
    def __init__(self, root, maximum, ttl):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.maximum = maximum
        self.ttl = ttl
        self.records = {}
        self.lock = Lock()

    def _remove(self, identifier, record, *, empty_only=False):
        """The registry lock is held; never remove a leased or locked session."""
        if record.active_requests or not record.lock.acquire(blocking=False):
            return False
        try:
            if empty_only and (record.files or any(record.state.values()) or any(record.directory.iterdir())):
                return False
            shutil.rmtree(record.directory, ignore_errors=True)
            del self.records[identifier]
            return True
        finally:
            record.lock.release()

    def get(self, identifier, *, create=False, csrf_token=None):
        """Lease an existing session, or explicitly create one for the session endpoint."""
        now = time.monotonic()
        with self.lock:
            for key, record in list(self.records.items()):
                if now - record.touched > self.ttl:
                    self._remove(key, record)
            if identifier not in self.records:
                if not create:
                    return None, None
                if len(self.records) >= self.maximum:
                    for key, record in sorted(self.records.items(), key=lambda item: item[1].touched):
                        if self._remove(key, record, empty_only=True):
                            break
                    else:
                        raise RequestError("All session slots are busy. Try again later.", 503)
                identifier = uuid.uuid4().hex
                directory = self.root / identifier
                directory.mkdir(mode=0o700)
                self.records[identifier] = UserSession(directory)
            record = self.records[identifier]
            if csrf_token is not None and not secrets.compare_digest(
                    csrf_token.encode("utf-8"), record.csrf.encode("utf-8")):
                raise RequestError("Session token is missing or expired. Refresh the page.", 403)
            record.touched = now
            record.active_requests += 1
            return identifier, record

    def release(self, record):
        with self.lock:
            record.active_requests -= 1
            record.touched = time.monotonic()


class RequestLimiter:
    def __init__(self):
        self.windows = {}
        self.lock = Lock()

    def check(self, key, limit):
        now = time.monotonic()
        with self.lock:
            for old_key, timestamps in list(self.windows.items()):
                if not timestamps or now - timestamps[-1] >= 60:
                    del self.windows[old_key]
            if key not in self.windows and len(self.windows) >= 2000:
                raise RequestError("The service is busy. Try again shortly.", 429)
            timestamps = self.windows.setdefault(key, deque())
            while timestamps and now - timestamps[0] >= 60:
                timestamps.popleft()
            if len(timestamps) >= limit:
                raise RequestError("Too many requests. Wait a minute and try again.", 429)
            timestamps.append(now)


def _metadata(service):
    if service is None:
        return {}
    return getattr(service, "metadata", getattr(service, "bundle", {}).get("metadata", {}))


def _inspect_csv(path, service):
    """Use the tool service's shared schema and limits; preserve invalid rows."""
    try:
        inspected = service.inspect_csv(path)
        return {"rows": inspected["total_count"], "columns": inspected["columns"]}
    except ValueError as exc:
        raise RequestError(str(exc)) from exc


def _disk_usage(directory):
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def create_app(config=None, *, service=None, agent=None):
    project = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(project / "templates"), static_folder=str(project / "static"))
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY"),
        MODEL_PATH=os.environ.get("MODEL_PATH") or str(project / "models/production.joblib"),
        SESSION_ROOT=None,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
        TRUST_PROXY=os.environ.get("TRUST_PROXY", "false").lower() == "true",
        MAX_UPLOAD_BYTES=5 * 1024 * 1024,
        MAX_CONTENT_LENGTH=5 * 1024 * 1024 + 65536,
        MAX_CSV_ROWS=10000,
        MAX_CELL_LENGTH=4096,
        MAX_FILES_PER_SESSION=5,
        MAX_SESSION_BYTES=25 * 1024 * 1024,
        MAX_SESSIONS=100,
        SESSION_TTL_SECONDS=3600,
        REQUESTS_PER_MINUTE=60,
        CHAT_REQUESTS_PER_MINUTE=10,
        MAX_MESSAGE_LENGTH=4000,
        MAX_TOOL_CALLS=6,
    )
    if config:
        app.config.update(config)
    if app.config["TRUST_PROXY"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    if not app.config["SECRET_KEY"]:
        if app.config.get("TESTING"):
            app.config["SECRET_KEY"] = secrets.token_hex(32)
        else:
            raise RuntimeError("FLASK_SECRET_KEY must be set in the environment.")

    if service is None:
        try:
            import joblib
            from .tools import ToolService
            service = ToolService(joblib.load(app.config["MODEL_PATH"]))
        except (OSError, ValueError, ImportError, KeyError, TypeError):
            app.logger.exception("Production model could not be loaded")
    if service is not None and hasattr(service, "configure_limits"):
        service.configure_limits(max_bytes=app.config["MAX_UPLOAD_BYTES"],
                                 max_rows=app.config["MAX_CSV_ROWS"],
                                 max_cell_length=app.config["MAX_CELL_LENGTH"])
    if agent is None and service is not None:
        from .agent import Agent
        agent = Agent(service, max_tool_calls=app.config["MAX_TOOL_CALLS"])
    root = app.config["SESSION_ROOT"] or tempfile.mkdtemp(prefix="ml-agent-sessions-")
    store = SessionStore(root, app.config["MAX_SESSIONS"], app.config["SESSION_TTL_SECONDS"])
    limiter = RequestLimiter()
    app.extensions["tool_service"] = service
    app.extensions["chat_agent"] = agent
    app.extensions["session_store"] = store

    def current_session(*, create=False, csrf_token=None, missing_status=403):
        if "leased_session" in g:
            return g.leased_session
        identifier, record = store.get(session.get("sid"), create=create, csrf_token=csrf_token)
        if record is None:
            message = ("This result is unavailable in your session." if missing_status == 404 else
                       "Session token is missing or expired. Refresh the page.")
            raise RequestError(message, missing_status)
        session["sid"] = identifier
        g.leased_session = identifier, record
        return identifier, record

    @app.teardown_request
    def release_session(_error):
        leased = g.pop("leased_session", None)
        if leased is not None:
            store.release(leased[1])

    @app.before_request
    def guard_request():
        if request.path.startswith("/static/") or request.path == "/health":
            return None
        limiter.check(("ip", request.remote_addr or "unknown"), app.config["REQUESTS_PER_MINUTE"])
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            for header in ("Origin", "Referer"):
                supplied = request.headers.get(header)
                if supplied:
                    origin = urlsplit(supplied)
                    expected = urlsplit(request.host_url)
                    if (origin.scheme, origin.netloc) != (expected.scheme, expected.netloc):
                        raise RequestError("This request must come from the same website.", 403)
            token = request.headers.get("X-CSRF-Token", "")
            current_session(csrf_token=token)

    @app.after_request
    def secure_response(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestError)
    def request_error(exc):
        return jsonify(error=str(exc)), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        metadata = _metadata(service)
        available = service is not None
        return jsonify(
            status="ok" if available else "unavailable",
            model_version=metadata.get("model_version"),
            selected_model=metadata.get("selected_model"),
            openai_configured=bool(os.environ.get("OPENAI_API_KEY")),
            commit=os.environ.get("RENDER_GIT_COMMIT", "local"),
        ), 200 if available else 503

    @app.get("/api/session")
    def session_details():
        _, record = current_session(create=True)
        with record.lock:
            files = [{"id": key, "name": value["name"], "rows": value["rows"], "columns": value["columns"]}
                     for key, value in record.files.items()]
            return jsonify(csrf_token=record.csrf, files=files,
                           results=record.state.get("results", [])[-20:],
                           activity=record.state.get("activity", [])[-100:])

    @app.post("/api/upload")
    def upload():
        if service is None:
            raise RequestError("The production model is unavailable. Uploads cannot be validated.", 503)
        if "file" not in request.files:
            raise RequestError("Choose a CSV file to upload.")
        incoming = request.files["file"]
        if not incoming.filename or Path(incoming.filename).suffix.lower() != ".csv":
            raise RequestError("Only CSV files are accepted.")
        if incoming.mimetype not in {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream", "text/plain"}:
            raise RequestError("Only CSV files are accepted.")
        _, record = current_session()
        with record.lock:
            if len(record.files) >= app.config["MAX_FILES_PER_SESSION"]:
                raise RequestError("The session file limit is reached. Reset the session to upload more.", 413)
            content = incoming.stream.read(app.config["MAX_UPLOAD_BYTES"] + 1)
            if len(content) > app.config["MAX_UPLOAD_BYTES"]:
                raise RequestError("CSV upload exceeds the size limit.", 413)
            if _disk_usage(record.directory) + len(content) > app.config["MAX_SESSION_BYTES"]:
                raise RequestError("The session storage limit is reached. Reset the session to continue.", 413)
            identifier = uuid.uuid4().hex
            path = record.directory / f"{identifier}.csv"
            path.write_bytes(content)
            try:
                details = _inspect_csv(path, service)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            name = secure_filename(incoming.filename)[:160] or "upload.csv"
            record.files[identifier] = {"path": path, "name": name, **details}
            return jsonify(file={"id": identifier, "name": name, **details}), 201

    @app.post("/api/chat")
    def chat():
        if agent is None:
            raise RequestError("The production model is unavailable. Chat is temporarily unavailable.", 503)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
            raise RequestError("Send a JSON message containing text.")
        message = payload["message"].strip()
        if not message or len(message) > app.config["MAX_MESSAGE_LENGTH"]:
            raise RequestError(f"Message must contain 1–{app.config['MAX_MESSAGE_LENGTH']} characters.")
        identifier, record = current_session()
        limiter.check(("chat", identifier), app.config["CHAT_REQUESTS_PER_MINUTE"])
        with record.lock:
            if _disk_usage(record.directory) >= app.config["MAX_SESSION_BYTES"]:
                raise RequestError("The session storage limit is reached. Reset the session to continue.", 413)
            previous_paths = set(record.directory.iterdir())
            previous_state = deepcopy(record.state)

            def rollback():
                for path in set(record.directory.iterdir()) - previous_paths:
                    if path.is_file():
                        path.unlink(missing_ok=True)
                record.state.clear()
                record.state.update(previous_state)

            try:
                response = agent.chat(message, record.state, record.files)
                if _disk_usage(record.directory) > app.config["MAX_SESSION_BYTES"]:
                    raise RequestError("The results exceed the session storage limit. Use a smaller file.", 413)
            except RequestError:
                rollback()
                raise
            except Exception:
                rollback()
                app.logger.exception("Chat failed")
                raise RequestError("The chat service failed. Your files are still available; try again.", 502)
            # Retention becomes destructive only after the turn passes quotas.
            # Until then rollback can restore every prior download and its file.
            retained_downloads = {Path(path) for path in record.state.get("downloads", {}).values()}
            upload_paths = {Path(file["path"]) for file in record.files.values()}
            old_downloads = {Path(path) for path in previous_state.get("downloads", {}).values()}
            for path in old_downloads - retained_downloads - upload_paths:
                if path.parent.resolve() == record.directory.resolve() and path.is_file():
                    path.unlink(missing_ok=True)
            return jsonify(response)

    @app.get("/api/download/<identifier>")
    def download(identifier):
        _, record = current_session(missing_status=404)
        with record.lock:
            path = record.state.get("downloads", {}).get(identifier)
            if path is None:
                raise RequestError("This result is unavailable in your session.", 404)
            path = Path(path)
            if path.parent.resolve() != record.directory.resolve() or not path.is_file():
                raise RequestError("This result is unavailable in your session.", 404)
            return send_file(path, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

    @app.post("/api/reset")
    def reset():
        _, record = current_session()
        with record.lock:
            for path in record.directory.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
            record.files.clear()
            record.state.clear()
            record.state.update(history=[], results=[], activity=[], downloads={})
            return jsonify(status="ok")

    return app
