import os
import subprocess
import sys
import time


def start_backend():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=os.getcwd(),
        env=env,
    )


def start_frontend():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            "3000",
        ],
        cwd=os.path.join(os.getcwd(), "frontend"),
        env=env,
    )


if __name__ == "__main__":
    backend = start_backend()
    time.sleep(2)
    frontend = start_frontend()
    print("Backend running on http://localhost:8000")
    print("Frontend running on http://localhost:3000")
    print("Press Ctrl+C to stop both services.")
    try:
        backend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()
        print("Stopped backend and frontend.")
