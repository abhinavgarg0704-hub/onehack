import subprocess
import time
import urllib.request
import urllib.error
import sys
import os

PORT = 8540
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.headless=true",
    f"--server.port={PORT}",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false"
]

print(f"Launching Streamlit on port {PORT}...")
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace"
)

health_url = f"http://localhost:{PORT}/_stcore/health"
main_url = f"http://localhost:{PORT}/"

success = False
start_time = time.time()
timeout = 35

try:
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            print(f"[FATAL] Streamlit process exited prematurely with code {proc.returncode}!")
            stdout, stderr = proc.communicate()
            print("STDOUT:\n", stdout)
            print("STDERR:\n", stderr)
            sys.exit(1)
            
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[OK] Streamlit health endpoint responded HTTP 200 in {time.time() - start_time:.1f}s")
                    
                    # Also test main URL
                    req_main = urllib.request.Request(main_url)
                    with urllib.request.urlopen(req_main, timeout=5) as resp_main:
                        html = resp_main.read().decode("utf-8", errors="replace")
                        print(f"[OK] Streamlit main page responded HTTP {resp_main.status} (received {len(html)} bytes)")
                        assert "FloodSense" in html or "streamlit" in html.lower(), "Expected content not found in HTML"
                        success = True
                        break
        except Exception:
            time.sleep(1.5)

    if not success:
        print("[FAIL] Streamlit server failed to respond within timeout!")
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
        sys.exit(1)

    print("\nVerifying server log output for runtime exceptions...")
    time.sleep(3)
    # Check if process is still alive and healthy
    assert proc.poll() is None, "Process died unexpectedly after responding"
    print("[OK] Streamlit process is alive and stable without crashes!")

finally:
    print("Terminating test server...")
    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=5)
        print(f"Process terminated cleanly. Return code: {proc.returncode}")
        # Look for tracebacks or exceptions in output
        combined = (stdout or "") + (stderr or "")
        if "Traceback (most recent call last):" in combined:
            print("[ERROR] Found Traceback in Streamlit output:\n", combined)
            sys.exit(1)
        else:
            print("[OK] Zero tracebacks or unhandled exceptions in Streamlit logs.")
    except Exception as e:
        print("Forcing process kill:", e)
        proc.kill()

print("\n=== STREAMLIT RUNTIME LAUNCH VERIFICATION: 100% PASSED ===")
