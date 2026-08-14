import re
import subprocess
import sys
import time

try:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def start_tunnel():
    print("=== Starting Global Public Tunnel to Finance RAG App ===")
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", "80:localhost:8000",
        "nokey@localhost.run",
    ]
    
    while True:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line_str = line.strip()
                match = re.search(r"https://[a-zA-Z0-9\-]+\.lhr\.life", line_str)
                if match:
                    url = match.group(0)
                    print("\n" + "=" * 60)
                    print(f"🎉 LIVE GLOBAL PUBLIC URL: {url}")
                    print("=" * 60 + "\n", flush=True)
                else:
                    if line_str and not line_str.startswith("Pseudo-terminal"):
                        print(f"[tunnel] {line_str}", flush=True)
            proc.wait()
        except Exception as e:
            print(f"[tunnel error] {e}", flush=True)
        print("[tunnel] Reconnecting in 3 seconds...", flush=True)
        time.sleep(3)

if __name__ == "__main__":
    start_tunnel()
