#!/usr/bin/env python3
"""
Native desktop launcher for the UFC ML Streamlit app.

Run from source with:
    .venv\\Scripts\\python.exe ufc_desktop_app.py
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import traceback

import requests
from streamlit import development as streamlit_development
from streamlit import config as streamlit_config
from streamlit.web import bootstrap
import webview

# Import app modules so PyInstaller bundles them into the desktop executable.
import betting_utils as _betting_utils  # noqa: F401
import build_profile_aligned_dataset as _dataset_builder  # noqa: F401
import process_ufc_data as _trainer  # noqa: F401
import run_pipeline as _pipeline  # noqa: F401
import ufc_fight_predictor as _predictor  # noqa: F401
import ufc_ml_ui as _ui  # noqa: F401
import ufc_profile_schema as _schema  # noqa: F401


APP_TITLE = "UFC ML Desktop"
HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 60
LOG_FILE_NAME = "ufc_desktop_app.log"

LOADING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>UFC ML Desktop</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(1000px 500px at 85% -10%, rgba(255, 0, 0, 0.18), transparent 50%),
        radial-gradient(900px 500px at -10% 10%, rgba(255, 0, 0, 0.12), transparent 50%),
        #090909;
      color: #f3f3f3;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .card {
      width: min(560px, calc(100vw - 48px));
      padding: 28px 32px;
      border: 1px solid #2b2b2b;
      border-radius: 14px;
      background: linear-gradient(135deg, #0f0f0f 0%, #171717 60%, #260707 100%);
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }
    .eyebrow {
      color: #ff4b43;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0 0 10px 0;
      font-size: 38px;
      line-height: 1;
      text-transform: uppercase;
    }
    p {
      margin: 0;
      color: #cccccc;
      font-size: 18px;
    }
    .bar {
      margin-top: 22px;
      width: 100%;
      height: 4px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar::after {
      content: "";
      display: block;
      width: 35%;
      height: 100%;
      background: linear-gradient(90deg, #e10600, #ff5a52);
      animation: loading 1.2s ease-in-out infinite;
    }
    @keyframes loading {
      from { transform: translateX(-100%); }
      to { transform: translateX(300%); }
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="eyebrow">Desktop Launch</div>
    <h1>UFC ML</h1>
    <p>Starting the local app server and loading the interface.</p>
    <div class="bar"></div>
  </div>
</body>
</html>
"""


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def normalize_runtime_directory() -> Path:
    root = app_root()
    os.chdir(root)
    return root


def log_path() -> Path:
    return app_root() / LOG_FILE_NAME


def reset_log() -> None:
    log_path().write_text("", encoding="utf-8")


def log_message(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path().open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def attach_window_logging(window: webview.Window) -> None:
    def on_initialized(renderer: str) -> None:
        log_message(f"Webview renderer initialized: {renderer}")

    def on_request(request) -> None:
        log_message(f"Request sent: {request.method} {request.url}")

    def on_response(response) -> None:
        log_message(f"Response received: {response.status_code} {response.url}")

    window.events.initialized += on_initialized
    window.events.request_sent += on_request
    window.events.response_received += on_response


def build_entry_script() -> Path:
    wrapper_dir = Path(tempfile.gettempdir()) / "ufc_ml_desktop"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / "streamlit_entry.py"
    wrapper_path.write_text("from ufc_ml_ui import main\nmain()\n", encoding="utf-8")
    return wrapper_path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def run_streamlit_server(
    entry_script: Path,
    port: int,
    server_state: dict[str, str],
) -> None:
    try:
        log_message(f"Starting embedded Streamlit server on {HOST}:{port}")
        streamlit_config.set_option("global.developmentMode", False)
        streamlit_development.is_development_mode = False
        streamlit_config.set_option("server.headless", True)
        streamlit_config.set_option("server.address", HOST)
        streamlit_config.set_option("server.port", int(port))
        streamlit_config.set_option("server.fileWatcherType", "none")
        streamlit_config.set_option("server.runOnSave", False)
        streamlit_config.set_option("browser.gatherUsageStats", False)
        streamlit_config.set_option("logger.hideWelcomeMessage", True)
        streamlit_config.set_option("browser.serverAddress", HOST)
        streamlit_config.set_option("browser.serverPort", int(port))
        original_signal_handler = bootstrap._set_up_signal_handler
        bootstrap._set_up_signal_handler = lambda _server: None
        try:
            bootstrap.run(
                str(entry_script),
                False,
                [],
                {},
            )
        finally:
            bootstrap._set_up_signal_handler = original_signal_handler
    except Exception:
        server_state["error"] = traceback.format_exc()
        log_message("Embedded Streamlit server crashed:")
        log_message(server_state["error"].rstrip())


def start_server_thread(
    entry_script: Path,
    port: int,
    server_state: dict[str, str],
) -> threading.Thread:
    server_thread = threading.Thread(
        target=run_streamlit_server,
        args=(entry_script, port, server_state),
        daemon=True,
        name="ufc-ml-streamlit-server",
    )
    server_thread.start()
    return server_thread


def wait_for_server(
    port: int,
    server_thread: threading.Thread,
    server_state: dict[str, str],
    timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
) -> str:
    app_url = f"http://{HOST}:{port}"
    health_url = f"{app_url}/_stcore/health"
    root_url = f"{app_url}/"
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        if server_state.get("error"):
            raise RuntimeError(
                "Embedded app server crashed during startup.\n\n"
                f"{server_state['error']}"
            )
        if not server_thread.is_alive():
            raise RuntimeError(
                "Embedded app server exited before the desktop window finished loading."
            )
        try:
            health_response = requests.get(health_url, timeout=1)
            if health_response.ok:
                root_response = requests.get(root_url, timeout=1)
                if root_response.status_code == 200:
                    log_message(f"Embedded Streamlit server is healthy at {app_url}")
                    log_message(f"Embedded Streamlit root returned 200 at {root_url}")
                    return app_url
                log_message(
                    f"Embedded Streamlit root not ready yet: "
                    f"{root_response.status_code} {root_url}"
                )
        except Exception as exc:  # pragma: no cover - network timing varies
            last_error = exc
        time.sleep(0.25)

    raise RuntimeError(
        f"Local app server did not start within {timeout_seconds} seconds."
    ) from last_error


def build_error_html(message: str) -> str:
    safe_message = html.escape(message)
    safe_log_path = html.escape(str(log_path()))
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>UFC ML Desktop Error</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #090909;
      color: #f3f3f3;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    .card {{
      width: min(720px, calc(100vw - 48px));
      padding: 28px 32px;
      border: 1px solid #4a1111;
      border-radius: 14px;
      background: linear-gradient(135deg, #160909 0%, #1d0c0c 100%);
    }}
    h1 {{
      margin: 0 0 12px 0;
      color: #ff5a52;
      font-size: 34px;
      text-transform: uppercase;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(255, 255, 255, 0.04);
      border-radius: 10px;
      padding: 14px 16px;
      margin-top: 16px;
      color: #f0d9d8;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Desktop Launch Failed</h1>
    <div>The app server could not start correctly.</div>
    <div>See the startup log at: {safe_log_path}</div>
    <pre>{safe_message}</pre>
  </div>
</body>
</html>
"""


def force_exit() -> None:
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch UFC ML desktop app")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start the embedded Streamlit server and exit once the health check passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_runtime_directory()
    reset_log()
    log_message(f"Launching {APP_TITLE} from {app_root()}")

    entry_script = build_entry_script()
    port = find_free_port()
    server_state: dict[str, str] = {}
    server_thread = start_server_thread(entry_script, port, server_state)

    if args.smoke_test:
        ready_url = wait_for_server(port, server_thread, server_state)
        root_response = requests.get(ready_url, timeout=5)
        print(f"Desktop app server ready on {ready_url}")
        print(f"Desktop app root status: {root_response.status_code}")
        return

    window_kwargs = {
        "title": APP_TITLE,
        "width": 1500,
        "height": 960,
        "min_size": (1100, 760),
        "background_color": "#090909",
        "text_select": True,
    }

    try:
        ready_url = wait_for_server(port, server_thread, server_state)
        ready_url = f"{ready_url}/"
        log_message(f"Opening desktop window at {ready_url}")
        window = webview.create_window(
            url=ready_url,
            **window_kwargs,
        )
    except Exception as exc:
        log_message(f"Desktop launch failed: {exc}")
        window = webview.create_window(
            html=build_error_html(str(exc)),
            **window_kwargs,
        )
    attach_window_logging(window)
    window.events.closed += force_exit
    webview.start(gui="edgechromium")
    force_exit()


if __name__ == "__main__":
    main()
