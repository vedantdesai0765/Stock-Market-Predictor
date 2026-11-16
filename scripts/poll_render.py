import time
import json
import urllib.request
import urllib.error

url = "https://stock-predictor-api-5wwr.onrender.com/api/v1/health"
print(f"Monitoring Render backend at {url}...", flush=True)

start = time.time()
while time.time() - start < 360:
    elapsed = int(time.time() - start)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AntigravityMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"\n[+] Render responded after {elapsed}s:", flush=True)
            print(json.dumps(data, indent=2), flush=True)
            if data.get("registered_stocks") == 30:
                print("[+] SUCCESS: registered_stocks == 30!", flush=True)
                break
            else:
                print(f"[i] registered_stocks is currently {data.get('registered_stocks')}. Waiting for new deploy to take over...", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[{elapsed}s] HTTP Error {e.code}: {e.reason}", flush=True)
    except urllib.error.URLError as e:
        print(f"[{elapsed}s] Connection waiting ({e.reason})", flush=True)
    except TimeoutError:
        print(f"[{elapsed}s] Read timed out (Render building/waking)", flush=True)
    except Exception as e:
        print(f"[{elapsed}s] {type(e).__name__}: {e}", flush=True)
    time.sleep(12)
