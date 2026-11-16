import urllib.request
import json
import sys

BASE_URL = "https://stock-predictor-api-5wwr.onrender.com"

endpoints = [
    ("/api/v1/health", "GET", None),
    ("/api/v1/stocks", "GET", None),
    ("/api/v1/stocks/ICICIBANK", "GET", None),
    ("/api/v1/stocks/BHARTIARTL/data-status", "GET", None),
    ("/api/v1/stocks/MARUTI/prediction", "GET", None),
    ("/api/v1/stocks/TITAN/forecast", "GET", None),
    ("/api/v1/stocks/CIPLA/signals/latest", "GET", None),
    ("/api/v1/portfolio/preview", "POST", {"ticker": "ICICIBANK", "side": "BUY", "quantity": 5}),
]

print(f"Testing Render Production Backend ({BASE_URL})...\n")

all_ok = True
for path, method, body in endpoints:
    url = f"{BASE_URL}{path}"
    try:
        headers = {"User-Agent": "RenderVerifier/1.0"}
        data = None
        if body:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = json.loads(resp.read().decode("utf-8"))
            print(f"[HTTP {resp.status}] {method} {path}")
            if path == "/api/v1/health":
                print(f"  -> status: {content.get('status')}, registered_stocks: {content.get('registered_stocks')}, db: {content.get('database')}")
            elif path == "/api/v1/stocks":
                total = content.get('total')
                sample = [s.get('ticker') for s in content.get('stocks', [])[:5]]
                print(f"  -> total stocks: {total}, sample: {sample}")
            elif path == "/api/v1/stocks/ICICIBANK":
                print(f"  -> ticker: {content.get('ticker')}, name: {content.get('name')}, sector: {content.get('sector')}, has_model: {content.get('has_model')}")
            elif "data-status" in path:
                print(f"  -> ticker: {content.get('ticker')}, raw_max_date: {content.get('raw_dataset', {}).get('max_date')}, rows: {content.get('raw_dataset', {}).get('rows')}")
            elif "prediction" in path:
                print(f"  -> ticker: {content.get('ticker')}, signal: {content.get('signal')}, prob_up: {content.get('prob_up')}, threshold: {content.get('operating_threshold')}")
            elif "forecast" in path:
                print(f"  -> ticker: {content.get('ticker')}, top_model: {content.get('top_model', {}).get('name')}, val_acc: {content.get('top_model', {}).get('validation_accuracy')}")
            elif "signals/latest" in path:
                print(f"  -> ticker: {content.get('ticker')}, date: {content.get('date')}, signal: {content.get('signal')}, price: {content.get('close_price')}")
            elif "preview" in path:
                print(f"  -> ticker: {content.get('ticker')}, execution_price: {content.get('execution_price')}, net_amount: {content.get('net_amount')}, is_valid: {content.get('is_valid')}")
    except Exception as e:
        all_ok = False
        print(f"[FAILED] {method} {path}: {e}")

if all_ok:
    print("\nALL PRODUCTION RENDER ENDPOINTS VERIFIED SUCCESSFULLY!")
else:
    print("\nSOME ENDPOINTS ENCOUNTERED ISSUES.")
