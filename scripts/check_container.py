"""Exercise the runtime Docker image under Render's 512 MiB memory budget.

Only the OpenAI HTTP provider is stubbed. The image's default Gunicorn command,
Flask application, frozen model, CSV tools and downloads run without substitution.
Uses the Python standard library and the Docker CLI; no development dependencies.
"""
from __future__ import annotations

import argparse
import csv
from http.cookies import SimpleCookie
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


ROOT = Path(__file__).resolve().parents[1]
PROBE_COMMIT = "container-compatibility-probe"
MEMORY_BYTES = 512 * 1024 * 1024

# This server routes only the probe's fixed prompts and supplies result references.
# It never predicts, supplies metrics, imports application code, or receives CSVs.
FAKE_PROVIDER = r'''import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import uuid

calls = []

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.respond(200, {"provider": "mocked", "request_count": len(calls),
                           "tool_calls": [name for name in calls if name != "final"]})

    def do_POST(self):
        try:
            if self.path != "/v1/responses":
                raise ValueError("Unexpected provider path")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inputs = body["input"]
            encoded = json.dumps(inputs)
            if any(word in encoded for word in ("ImportedSymbols", "BaseOfCode", "kernel32.dll", "/tmp/ml-agent-sessions-")):
                raise ValueError("CSV content or local paths were sent to provider")
            if inputs[-1].get("type") == "function_call_output":
                result = json.loads(inputs[-1]["output"])
                text = json.dumps({"kind": "results", "result_ids": [result["result_id"]], "focus": "summary"})
                output = [{"id": "msg-" + uuid.uuid4().hex, "type": "message", "role": "assistant",
                           "status": "completed", "content": [{"type": "output_text", "annotations": [], "text": text}]}]
                calls.append("final")
            else:
                prompt = next(item["content"] for item in reversed(inputs) if item.get("role") == "user")
                files = re.findall(r"file ([a-f0-9]{32})", prompt)
                context = json.loads(inputs[0]["content"].split(": ", 1)[1])
                registered = {item["file_id"] for item in context["registered_files"]}
                if not files or any(identifier not in registered for identifier in files):
                    raise ValueError("Unregistered file reference")
                if "only if accuracy >=" in prompt:
                    name = "evaluate_then_predict"
                    threshold = re.search(r"accuracy >= ([0-9.]+)", prompt)
                    args = {"evaluation_file_id": files[0], "prediction_file_id": files[1],
                            "row_index": 0, "min_accuracy": float(threshold.group(1))}
                elif prompt.startswith("Classify every row"):
                    name, args = "predict_batch", {"file_id": files[0]}
                elif prompt.startswith("Classify row 0"):
                    name, args = "predict_single", {"file_id": files[0], "row_index": 0}
                elif prompt.startswith("Evaluate file"):
                    name, args = "evaluate", {"file_id": files[0]}
                else:
                    raise ValueError("Unexpected probe prompt")
                if name not in {tool["name"] for tool in body["tools"]}:
                    raise ValueError("Tool was unavailable")
                call_id = "call-" + uuid.uuid4().hex
                output = [{"id": call_id, "type": "function_call", "call_id": call_id,
                           "name": name, "status": "completed", "arguments": json.dumps(args)}]
                calls.append(name)
            self.respond(200, {"id": "resp-" + uuid.uuid4().hex, "object": "response", "created_at": 1,
                               "model": body["model"], "status": "completed", "output": output,
                               "parallel_tool_calls": False, "error": None, "incomplete_details": None,
                               "instructions": None, "metadata": {}, "tools": [], "tool_choice": "auto"})
        except Exception:
            self.respond(400, {"error": {"message": "Container probe provider contract failed",
                                       "type": "invalid_request_error", "param": None, "code": "probe_contract"}})

ThreadingHTTPServer(("127.0.0.1", 18081), Handler).serve_forever()
'''


class ProbeError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def docker(*arguments, environment=None):
    completed = subprocess.run(["docker", *arguments], capture_output=True, text=True,
                               env=environment, timeout=120)
    if completed.returncode:
        raise ProbeError(f"Docker {arguments[0]} failed: {completed.stderr.strip()[:800]}")
    return completed.stdout.strip()


class Browser:
    """Forward a secure session cookie explicitly over the local proxy connection."""
    def __init__(self, base, statuses):
        self.base = base
        self.statuses = statuses
        self.cookie = ""
        self.token = None

    def request(self, method, path, *, payload=None, content_type=None, expected=200, label=None):
        headers = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.77"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        if method != "GET":
            headers["Origin"] = self.base.replace("http://", "https://", 1)
            headers["X-CSRF-Token"] = self.token or ""
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(self.base + path, data=payload, headers=headers, method=method)
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            response = error
        with response:
            status = response.status
            contents = response.read()
            for value in response.headers.get_all("Set-Cookie", []):
                cookies = SimpleCookie()
                cookies.load(value)
                if "session" in cookies:
                    cookie = cookies["session"]
                    require(bool(cookie["secure"]) and bool(cookie["httponly"]), "Session cookie must be secure and HttpOnly")
                    require(cookie["samesite"].lower() == "lax", "Session cookie SameSite must be Lax")
                    self.cookie = "session=" + cookie.value
        if label:
            self.statuses[label] = status
        require(status == expected, f"{label or path} returned HTTP {status}, expected {expected}")
        return contents

    def session(self, label):
        result = json.loads(self.request("GET", "/api/session", label=label))
        self.token = result["csrf_token"]
        require(bool(self.cookie), "Session endpoint did not issue a cookie")
        return result

    def upload(self, path, label):
        boundary = "probe-" + uuid.uuid4().hex
        prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                  "Content-Type: text/csv\r\n\r\n").encode()
        payload = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        result = json.loads(self.request("POST", "/api/upload", payload=payload,
                                        content_type=f"multipart/form-data; boundary={boundary}",
                                        expected=201, label=label))
        return result["file"]["id"]

    def chat(self, message, label, *, tool_error=False):
        result = json.loads(self.request("POST", "/api/chat", payload=json.dumps({"message": message}).encode(),
                                        content_type="application/json", label=label))
        require(result.get("error") == ("tool" if tool_error else None), f"{label} did not complete its expected tool path")
        return result


def tool_result(response, tool):
    matches = [result for result in response["results"] if result["tool"] == tool]
    require(len(matches) == 1, f"Expected one {tool} result")
    return matches[0]


def verify_download(browser, result, label):
    contents = browser.request("GET", result["download_url"], label=label)
    rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8"))))
    require(len(rows) == result["total_count"], f"{label} silently dropped rows")
    require(sum(row["status"] == "invalid" for row in rows) == result["invalid_count"], f"{label} invalid rows mismatch")
    require([int(row["row_index"]) for row in rows] == list(range(len(rows))), f"{label} row indexes changed")
    for row in rows:
        if row["status"] == "valid":
            require(row["prediction"] in {"0", "1"}, f"{label} has invalid model labels")
            require(0 <= float(row["malware_probability"]) <= 1, f"{label} has invalid model probabilities")
    return len(rows)


def run_probe(image, cpus=None):
    statuses = {}
    report = {"status": "running", "image": image, "provider": "mocked", "model": "frozen production pipeline",
              "memory_limit_bytes": MEMORY_BYTES, "swap_limit_bytes": MEMORY_BYTES,
              "cpu_limit": cpus, "container_port": 10000, "http_statuses": statuses}
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    expected_comparison = json.loads((ROOT / "models/comparison.json").read_text())
    name = "ml-runtime-probe-" + uuid.uuid4().hex
    environment = os.environ.copy()
    environment["FLASK_SECRET_KEY"] = secrets.token_hex(32)
    environment["OPENAI_API_KEY"] = "container-probe-fake-key"
    try:
        require(cpus is None or (math.isfinite(cpus) and cpus > 0), "CPU limit must be a finite positive number")
        with tempfile.TemporaryDirectory(prefix="ml-runtime-probe-") as directory:
            temporary = Path(directory)
            temporary.chmod(0o755)
            (temporary / "fake_openai.py").write_text(FAKE_PROVIDER)
            with (ROOT / "samples/labeled.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                source, header = next(reader), reader.fieldnames
            large = temporary / "large.csv"
            with large.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader()
                for index in range(10000):
                    writer.writerow({**source, "SHA1": f"probe-row-{index}"})
            require(large.stat().st_size <= 5 * 1024 * 1024, "Generated CSV exceeds the upload limit")
            report["large_batch_upload_bytes"] = large.stat().st_size
            cpu_arguments = [] if cpus is None else [f"--cpus={cpus}"]
            docker("run", *cpu_arguments, "--detach", "--name", name, "--memory=512m", "--memory-swap=512m",
                   "--publish", "127.0.0.1::10000", "--env", "PORT=10000", "--env", "FLASK_SECRET_KEY",
                   "--env", "OPENAI_API_KEY", "--env", "OPENAI_BASE_URL=http://127.0.0.1:18081/v1",
                   "--env", "SESSION_COOKIE_SECURE=true", "--env", "TRUST_PROXY=true",
                   "--env", "GUNICORN_CMD_ARGS=--access-logfile -",
                   "--env", f"RENDER_GIT_COMMIT={PROBE_COMMIT}", "--mount",
                   f"type=bind,src={temporary},dst=/probe,readonly", "--mount",
                   f"type=bind,src={ROOT / 'samples'},dst=/probe-samples,readonly", image, environment=environment)
            inspection = json.loads(docker("inspect", name))[0]
            report["image_id"] = inspection["Image"]
            report["container_user"] = inspection["Config"]["User"]
            require(inspection["Config"]["User"] not in {"", "0", "root"}, "Runtime must run as a non-root user")
            require(inspection["HostConfig"]["Memory"] == MEMORY_BYTES, "Docker memory constraint was not applied")
            require(inspection["HostConfig"]["MemorySwap"] == MEMORY_BYTES, "Docker swap constraint was not applied")
            if cpus is not None:
                require(inspection["HostConfig"]["NanoCpus"] == int(cpus * 1_000_000_000), "Docker CPU constraint was not applied")
            host_port = inspection["NetworkSettings"]["Ports"]["10000/tcp"][0]["HostPort"]
            base = f"http://127.0.0.1:{host_port}"
            docker("exec", "--detach", name, "python", "/probe/fake_openai.py")
            browser = Browser(base, statuses)
            deadline = time.monotonic() + 90
            while True:
                try:
                    health = json.loads(browser.request("GET", "/health"))
                    break
                except (OSError, URLError, ProbeError):
                    if time.monotonic() >= deadline:
                        raise ProbeError("Runtime did not become healthy on PORT=10000")
                    state = json.loads(docker("inspect", "--format", "{{json .State}}", name))
                    diagnostic = {key: state[key] for key in ("Status", "ExitCode", "OOMKilled")}
                    require(state["Running"], "Runtime exited before becoming healthy: " + json.dumps(diagnostic))
                    time.sleep(0.25)
            require(health["commit"] == PROBE_COMMIT, "Health did not expose the configured commit")
            require(health["model_version"] == manifest["model_version"], "Health model version differs from frozen manifest")
            require(health["selected_model"] == "LightGBM", "Health selected an unexpected model")
            report["model_version"] = health["model_version"]
            report["health_commit"] = health["commit"]
            statuses["health"] = 200
            require(b"/static/app.js" in browser.request("GET", "/", label="ui"), "Chat UI did not render")
            browser.request("GET", "/static/app.js", label="static_js")
            browser.request("GET", "/static/style.css", label="static_css")
            dashboard = browser.request("GET", "/analytics", label="model_dashboard")
            require(b"/static/dashboard.js" in dashboard, "Model dashboard did not render")
            for section in (b'id="data-split"', b'id="cv-walkthrough"',
                            b'id="model-comparison"', b'id="final-test"'):
                require(section in dashboard, "Guided experiment section did not render")
            browser.request("GET", "/static/dashboard.js", label="dashboard_js")
            browser.request("GET", "/static/dashboard.css", label="dashboard_css")
            browser.request("GET", "/static/walkthrough.css", label="walkthrough_css")
            comparison = json.loads(browser.request("GET", "/api/model-comparison", label="model_comparison"))
            require(comparison == expected_comparison, "Container dashboard differs from recorded experiment results")
            require(comparison["model_version"] == health["model_version"], "Dashboard and production model versions differ")
            report["dashboard"] = {"model_count": len(comparison["models"]),
                                   "folds": comparison["folds"],
                                   "selected_model": comparison["selected_model"],
                                   "source": "models/comparison.json"}
            browser.session("session_a")
            files = {sample: browser.upload(ROOT / "samples" / sample, "upload_" + sample) for sample in
                     ["single.csv", "invalid-rows.csv", "labeled.csv", "single-class.csv", "missing-labels.csv"]}
            single_reply = browser.chat(f"Classify row 0 of file {files['single.csv']}.", "single_prediction")
            single = tool_result(single_reply, "predict_single")
            require(single["prediction"] in (0, 1) and 0 <= single["malware_probability"] <= 1, "Single prediction is invalid")
            require(single["model_version"] == manifest["model_version"], "Single prediction used an unexpected model")
            batch_reply = browser.chat(f"Classify every row of file {files['invalid-rows.csv']}.", "invalid_batch_prediction")
            batch = tool_result(batch_reply, "predict_batch")
            require((batch["total_count"], batch["valid_count"], batch["invalid_count"]) == (4, 3, 1), "Invalid batch counts changed")
            verify_download(browser, batch, "invalid_batch_download")
            evaluation = tool_result(browser.chat(f"Evaluate file {files['labeled.csv']}.", "labeled_evaluation"), "evaluate")
            require(evaluation["evaluated_count"] == 4 and evaluation["evaluation_coverage"] == 1, "Labeled evaluation coverage changed")
            require(0 <= evaluation["auc"] <= 1 and 0 <= evaluation["accuracy"] <= 1, "Evaluation metrics are invalid")
            require(sum(map(sum, evaluation["confusion_matrix"])) == 4, "Confusion matrix counts changed")
            one_class = tool_result(browser.chat(f"Evaluate file {files['single-class.csv']}.", "single_class_evaluation"), "evaluate")
            require(one_class["auc"] is None and one_class["auc_reason"], "Single-class AUC must be unavailable")
            missing = tool_result(browser.chat(f"Evaluate file {files['missing-labels.csv']}.", "missing_label_rows"), "evaluate")
            require(missing["missing_label_count"] == 1 and missing["evaluated_count"] == 3, "Missing labels were silently dropped")
            no_labels = browser.chat(f"Evaluate file {files['single.csv']}.", "missing_label_column", tool_error=True)
            require(no_labels["activity"][-1]["status"] == "error" and "Label" in no_labels["reply"], "Missing label column did not fail visibly")
            passed = browser.chat(f"Evaluate file {files['labeled.csv']}; only if accuracy >= 0.0, predict row 0 of file {files['single.csv']}.", "conditional_pass")
            require([(entry["tool"], entry["status"]) for entry in passed["activity"]] ==
                    [("evaluate", "success"), ("predict_single", "success")], "Conditional pass did not evaluate before predicting")
            other = Browser(base, statuses)
            require(not other.session("session_b")["files"], "A new session saw another user's uploads")
            other.request("GET", batch["download_url"], expected=404, label="cross_session_download")
            failing_file = other.upload(ROOT / "samples/conditional-fail.csv", "upload_conditional_fail")
            prediction_file = other.upload(ROOT / "samples/single.csv", "upload_session_b_single")
            skipped = other.chat(f"Evaluate file {failing_file}; only if accuracy >= 1.0, predict row 0 of file {prediction_file}.", "conditional_skip")
            require([(entry["tool"], entry["status"]) for entry in skipped["activity"]] ==
                    [("evaluate", "success"), ("predict_single", "skipped")], "Conditional threshold failure still predicted")
            require(len(skipped["results"]) == 1, "Skipped condition produced a prediction result")
            large_browser = Browser(base, statuses)
            large_browser.session("session_large")
            large_file = large_browser.upload(large, "upload_large_batch")
            large_result = tool_result(large_browser.chat(f"Classify every row of file {large_file}.", "large_batch_prediction"), "predict_batch")
            require((large_result["total_count"], large_result["valid_count"], large_result["invalid_count"]) == (10000, 10000, 0), "Maximum-row batch did not score every row")
            verify_download(large_browser, large_result, "large_batch_download")
            browser.request("GET", "/health", label="health_after_large_batch")
            state = json.loads(docker("inspect", "--format", "{{json .State}}", name))
            require(state["Running"] and not state["OOMKilled"], "Container failed under the memory constraint")
            statistics = json.loads(docker("stats", "--no-stream", "--format", "{{json .}}", name))
            report["memory_observation"] = {key: statistics[key] for key in ("MemUsage", "MemPerc", "PIDs")}
            report["memory_observation_note"] = "One post-request Docker sample; this is not a peak-memory measurement."
            report["counts"] = {"invalid_batch": {key: batch[key] for key in ("total_count", "valid_count", "invalid_count")},
                                "large_batch": {key: large_result[key] for key in ("total_count", "valid_count", "invalid_count")},
                                "evaluated_rows": evaluation["evaluated_count"], "missing_label_rows": missing["missing_label_count"]}
            provider = json.loads(docker("exec", name, "python", "-c",
                                         "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18081').read().decode())"))
            require(all(tool in provider["tool_calls"] for tool in ("predict_single", "predict_batch", "evaluate", "evaluate_then_predict")), "Provider stub did not exercise every tool route")
            report["mock_provider"] = provider
            report["status"] = "ok"
            return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        return report
    finally:
        try:
            cleanup = subprocess.run(["docker", "rm", "--force", name], capture_output=True, text=True, timeout=30)
            if cleanup.returncode and report["status"] == "ok":
                report.update(status="failed", error="The disposable probe container could not be removed")
        except (OSError, subprocess.TimeoutExpired):
            if report["status"] == "ok":
                report.update(status="failed", error="The disposable probe container could not be removed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already-built runtime Docker image tag")
    parser.add_argument("--cpus", type=float, help="Optional finite positive Docker CPU limit, such as 0.1")
    parser.add_argument("--output", type=Path, help="Also write the aggregate JSON report to this path")
    arguments = parser.parse_args()
    if arguments.cpus is not None and (not math.isfinite(arguments.cpus) or arguments.cpus <= 0):
        parser.error("--cpus must be a finite positive number")
    report = run_probe(arguments.image, arguments.cpus)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
