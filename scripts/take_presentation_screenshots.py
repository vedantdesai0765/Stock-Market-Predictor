"""
Automated screenshot generator for presentation readiness.
Uses headless Chrome DevTools Protocol (CDP) to capture full-page high-resolution
screenshots of the polished dashboard across multiple views and stocks.
"""
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import websockets

SCREENSHOTS = [
    {
        "url": "http://localhost:8501?stock=INFY",
        "output": "screenshots/dashboard_terminal_polished.png",
        "description": "Infosys (INFY) - Polished Prediction Terminal",
        "delay": 7,
    },
    {
        "url": "http://localhost:8501?stock=INFY&view=explain",
        "output": "screenshots/why_did_it_move_polished.png",
        "description": "Infosys (INFY) - Polished Why did it move? View",
        "delay": 8,
    },
    {
        "url": "http://localhost:8501?stock=RELIANCE",
        "output": "screenshots/dashboard_terminal_reliance.png",
        "description": "Reliance Industries (RELIANCE) - Prediction Terminal",
        "delay": 7,
    },
]

async def run_captures():
    temp_dir = tempfile.mkdtemp()
    chrome_proc = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        f"--user-data-dir={temp_dir}",
        "--disable-gpu",
        "--window-size=1600,2600",
        "about:blank",
    ])
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json")
        tabs = json.loads(req.read().decode())
        page_tab = [t for t in tabs if t.get("type") == "page"][0]
        ws_url = page_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=30 * 1024 * 1024) as ws:
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

            for item in SCREENSHOTS:
                print(f"Navigating to {item['url']} for {item['description']}...")
                await send_cmd("Page.navigate", {"url": item["url"]})
                await asyncio.sleep(item["delay"])

                res = await send_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                out_path = os.path.abspath(item["output"])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(res["result"]["data"]))
                print(f"Saved: {out_path} ({os.path.getsize(out_path):,} bytes)")

    finally:
        chrome_proc.terminate()

if __name__ == "__main__":
    asyncio.run(run_captures())
