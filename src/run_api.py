from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import REQUIRED_PACKAGES, REQUIREMENTS_PATH, ensure_project_root_on_path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _missing_packages() -> list[str]:
    missing = []
    for package_name in REQUIRED_PACKAGES + ["fastapi", "uvicorn", "python_multipart"]:
        module_name = "multipart" if package_name == "python_multipart" else package_name
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _install_requirements_if_needed() -> None:
    missing_packages = _missing_packages()
    if not missing_packages:
        return

    print("Missing backend dependencies detected. Installing requirements.txt...")
    install_command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    result = subprocess.run(install_command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Dependency installation failed with exit code {result.returncode}.")

    remaining_missing = _missing_packages()
    if remaining_missing:
        raise SystemExit(
            "Dependencies are still missing after installation: " + ", ".join(remaining_missing)
        )


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def _expected_venv_python() -> Path:
    return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    ensure_project_root_on_path()
    _install_requirements_if_needed()

    import uvicorn

    host = os.environ.get("FRAUD_API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("FRAUD_API_PORT", DEFAULT_PORT))
    expected_python = _expected_venv_python()

    print(f"Starting backend on http://{host}:{port}/api")
    print(f"Python executable: {sys.executable}")
    if expected_python.exists() and Path(sys.executable).resolve() != expected_python.resolve():
        print(f"Recommended interpreter: {expected_python}")

    if not _port_is_available(host, port):
        raise SystemExit(
            f"Port {port} is already in use on {host}. "
            "Stop the conflicting process or set FRAUD_API_PORT to a different port."
        )

    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
