"""
scripts/capture_morning_screenshots.py
Captures genuine Before (live Vercel 10-stock deployment) and After (local 30-stock Morning Call) screenshots using headless Chrome DevTools Protocol (CDP).
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
import websockets

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "reports" / "screenshots" / "morning_call_30stocks"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not Path(CHROME_PATH).exists():
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

async def capture_cdp(url, out_path, delay=6, window_size="1400,2600"):
    temp_dir = tempfile.mkdtemp()
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        f"--user-data-dir={temp_dir}",
        "--disable-gpu",
        f"--window-size={window_size}",
        "about:blank",
    ])
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        tabs = json.loads(req.read().decode())
        page_tab = [t for t in tabs if t.get("type") == "page"][0]
        ws_url = page_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            async def send_cmd(method, params=None):
                import random
                cmd_id = random.randint(100, 99999)
                msg = {"id": cmd_id, "method": method}
                if params:
                    msg["params"] = params
                await ws.send(json.dumps(msg))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == cmd_id:
                        return resp

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            print(f"Navigating to {url}...")
            await send_cmd("Page.navigate", {"url": url})
            await asyncio.sleep(delay)

            res = await send_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
            out_file = Path(out_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(base64.b64decode(res["result"]["data"]))
            print(f"Captured: {out_file} ({out_file.stat().st_size:,} bytes)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

async def capture_interactive(url, actions, delay=4, window_size="1400,2800"):
    temp_dir = tempfile.mkdtemp()
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        f"--user-data-dir={temp_dir}",
        "--disable-gpu",
        f"--window-size={window_size}",
        "about:blank",
    ])
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        tabs = json.loads(req.read().decode())
        page_tab = [t for t in tabs if t.get("type") == "page"][0]
        ws_url = page_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            async def send_cmd(method, params=None):
                import random
                cmd_id = random.randint(100, 99999)
                msg = {"id": cmd_id, "method": method}
                if params:
                    msg["params"] = params
                await ws.send(json.dumps(msg))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == cmd_id:
                        return resp

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            print(f"Navigating to {url}...")
            await send_cmd("Page.navigate", {"url": url})
            await asyncio.sleep(delay)

            for step_name, js_code, out_file, capture_full in actions:
                if js_code:
                    print(f"Executing step '{step_name}'...")
                    await send_cmd("Runtime.evaluate", {"expression": js_code})
                    await asyncio.sleep(1)

                res = await send_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": capture_full})
                out_path = Path(out_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(res["result"]["data"]))
                print(f"Captured: {out_path} ({out_path.stat().st_size:,} bytes)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

def wait_for_url(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Capture BEFORE screenshot from live production Vercel (if not already captured)
    live_url = "https://stock-market-predictor-rouge.vercel.app"
    before_img = SCREENSHOT_DIR / "01_before_10_stocks_live.png"
    if not before_img.exists():
        print(f"Capturing BEFORE screenshot from {live_url}...")
        try:
            asyncio.run(capture_cdp(live_url, before_img, delay=7))
        except Exception as exc:
            print(f"Warning: Failed to capture live before screenshot: {exc}")

    # 2. Launch local backend and frontend
    python_exe = sys.executable
    print("Starting local FastAPI backend...")
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.app.main:app", "--port", "8000"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Starting local Next.js frontend...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "start"],
        cwd=str(ROOT / "frontend"),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Waiting for local backend (http://127.0.0.1:8000/api/v1/health)...")
        if not wait_for_url("http://127.0.0.1:8000/api/v1/health", timeout=20):
            print("Backend failed to start in time.")
            return

        print("Waiting for local frontend (http://localhost:3000)...")
        if not wait_for_url("http://localhost:3000", timeout=30):
            print("Frontend failed to start in time.")
            return

        print("Local stack is UP! Capturing AFTER screenshots...")
        actions = [
            (
                "Full 30-Stock Dashboard",
                None,
                SCREENSHOT_DIR / "02_after_30_stocks_full_dashboard.png",
                True
            ),
            (
                "Sector Filter: Financial Services",
                """(() => {
                    const sel = document.querySelector('select[aria-label="Filter by sector"]');
                    if (sel) {
                        sel.value = "Financial Services";
                        sel.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                })()""",
                SCREENSHOT_DIR / "03_after_sector_filter_banking.png",
                False
            ),
            (
                "Search Filter: Tata",
                """(() => {
                    const sel = document.querySelector('select[aria-label="Filter by sector"]');
                    if (sel) {
                        sel.value = "ALL";
                        sel.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    const inp = document.querySelector('input[placeholder*="Search"]');
                    if (inp) {
                        inp.value = "Tata";
                        inp.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                })()""",
                SCREENSHOT_DIR / "04_after_search_filter.png",
                False
            )
        ]
        asyncio.run(capture_interactive("http://localhost:3000", actions, delay=4, window_size="1400,1200"))

        print("Screenshots captured successfully!")

    finally:
        print("Cleaning up local processes...")
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=3)
        except Exception:
            backend_proc.kill()

        try:
            # Kill node process started by npm
            subprocess.run(["taskkill", "/F", "/IM", "node.exe", "/T"], capture_output=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
