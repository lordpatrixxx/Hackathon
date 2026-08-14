import re
import subprocess
import sys
import time

try:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def start_cf():
    print("=== Launching Cloudflare Global Edge Tunnel for Finance RAG ===")
    cmd = [
        ".\\cloudflared.exe",
        "tunnel",
        "--protocol",
        "http2",
        "--url",
        "http://127.0.0.1:8000",
        "--no-autoupdate",
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    url_found = None
    for line in proc.stdout:
        line_str = line.strip()
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line_str)
        if match:
            url_found = match.group(0)
            print("\n" + "=" * 60)
            print(f"🌟 LIVE PUBLIC CLOUDFLARE URL: {url_found}")
            print("=" * 60 + "\n", flush=True)
        else:
            if "Registered tunnel connection" in line_str or "Connection" in line_str:
                print(f"[cloudflare] {line_str}", flush=True)
    proc.wait()

if __name__ == "__main__":
    start_cf()
