"""
scripts/deploy_live.py
Automated end-to-end deployment script for Render (Backend) and Vercel (Frontend).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_URL = "https://stock-predictor-api-5wwr.onrender.com"


def check_auth():
    print("[1/5] Checking authentication & deployment configuration...")
    vc_authed = False
    vc_user = None
    try:
        res = subprocess.run(["npx", "--yes", "vercel", "whoami"],
                             cwd=str(ROOT), capture_output=True, text=True, shell=True)
        out = (res.stdout + res.stderr).strip()
        if res.returncode == 0 and "Logged out" not in out:
            vc_user = [line for line in out.splitlines() if line and not line.startswith("Vercel")][0]
            vc_authed = True
            print(f"  [+] Vercel CLI: Authenticated as '{vc_user}'")
        else:
            print("  [-] Vercel CLI: Not authenticated (Logged out)")
    except Exception as exc:
        print(f"  [-] Vercel check failed: {exc}")

    render_hook = os.getenv("RENDER_DEPLOY_HOOK")
    if render_hook:
        print("  [+] Render Deploy Hook: Configured in environment")
    else:
        print("  [i] Render Deploy Hook: Not set in local environment (Git auto-deploy active)")

    return vc_authed, vc_user


def verify_render_backend(backend_url, timeout=300):
    backend_url = backend_url.rstrip("/")
    health_url = f"{backend_url}/api/v1/health"
    print(f"[2/5] Verifying Render backend health at {health_url}...")
    print("  Waiting for Render web service to respond (free tier may take 30-60s on cold start)...")

    start_time = time.time()
    last_status = None
    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "AntigravityDeploy/2.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "healthy":
                    print(f"\n  [+] Backend is LIVE and healthy! (took {elapsed}s)")
                    print(f"      Project: {data.get('project')}")
                    print(f"      Stocks registered: {data.get('registered_stocks')}")
                    print(f"      Database: {data.get('database')}")
                    return True
        except urllib.error.HTTPError as err:
            last_status = f"HTTP {err.code}"
        except urllib.error.URLError as err:
            last_status = f"Connection waiting: {err.reason}"
        except Exception as exc:
            last_status = str(exc)

        print(f"  -> Waiting for Render service... ({elapsed}s elapsed, last: {last_status})", end="\r")
        time.sleep(10)

    print(f"\n  [-] Warning: Backend health check did not pass within {timeout}s.")
    print(f"      Last recorded response: {last_status}")
    return False


def deploy_vercel(backend_url):
    backend_url = backend_url.rstrip("/")
    api_url = f"{backend_url}/api/v1"
    print(f"[3/5] Deploying Next.js frontend to Vercel (API: {api_url})...")

    cmd = [
        "npx", "--yes", "vercel", "deploy", "frontend",
        "--prod",
        "--yes",
        "--create-if-missing",
        "--project", "stock-market-predictor",
        "--build-env", f"NEXT_PUBLIC_API_URL={api_url}",
        "--env", f"NEXT_PUBLIC_API_URL={api_url}",
    ]
    print(f"  Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, shell=True)
    out = (res.stdout + res.stderr).strip()

    print("  Vercel deployment logs:")
    for line in out.splitlines():
        print(f"    {line}")

    vercel_url = None
    for line in out.splitlines():
        clean = line.strip()
        if clean.startswith("https://") and "vercel.app" in clean:
            vercel_url = clean
            break

    if not vercel_url:
        res_info = subprocess.run(["npx", "--yes", "vercel", "project", "ls"],
                                  cwd=str(ROOT), capture_output=True, text=True, shell=True)
        for line in (res_info.stdout + res_info.stderr).splitlines():
            if "stock-market-predictor" in line and "https://" in line:
                for token in line.split():
                    if token.startswith("https://"):
                        vercel_url = token.rstrip("/")
                        break

    if vercel_url:
        print(f"  [+] Frontend deployed successfully: {vercel_url}")
    else:
        print("  [-] Notice: Check Vercel dashboard for production domain.")

    return vercel_url


def trigger_deploy_hook(hook_url):
    print(f"Triggering Render deploy hook: {hook_url[:35]}...")
    try:
        req = urllib.request.Request(hook_url, method="POST", headers={"User-Agent": "AntigravityDeploy"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  [+] Deploy hook response code: {resp.status}")
            return True
    except Exception as exc:
        print(f"  [-] Failed to trigger deploy hook: {exc}")
        return False


def verify_application(backend_url, vercel_url):
    backend_url = backend_url.rstrip("/")
    print("[4/5] Verifying complete application connectivity...")

    # Test Backend Health
    try:
        req = urllib.request.Request(f"{backend_url}/api/v1/health", headers={"User-Agent": "AntigravityDeploy"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [+] Backend /health: {data.get('status')} (DB: {data.get('database')})")
    except Exception as exc:
        print(f"  [-] Backend /health check failed: {exc}")

    # Test Morning Call endpoint
    try:
        req = urllib.request.Request(f"{backend_url}/api/v1/morning-call", headers={"User-Agent": "AntigravityDeploy"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [+] Backend /morning-call status: {data.get('status')}")
    except Exception as exc:
        print(f"  [-] Backend /morning-call check failed: {exc}")

    # Test Vercel Frontend
    if vercel_url:
        try:
            req = urllib.request.Request(vercel_url, headers={"User-Agent": "AntigravityDeploy"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"  [+] Vercel website HTTP response: {resp.status}")
        except Exception as exc:
            print(f"  [-] Vercel homepage check failed: {exc}")

    print("\n[5/5] Deployment Overview:")
    print("====================================================")
    print(f"Backend URL:        {backend_url}")
    print(f"API Base URL:       {backend_url}/api/v1")
    print(f"Interactive Docs:   {backend_url}/docs")
    print(f"Health Endpoint:    {backend_url}/api/v1/health")
    if vercel_url:
        print(f"Website URL:        {vercel_url}")
    print("====================================================")


def main():
    parser = argparse.ArgumentParser(description="Deploy Stock Market Predictor to Render and Vercel.")
    parser.add_argument("--backend-url", default=DEFAULT_RENDER_URL, help="Render backend URL")
    parser.add_argument("--hook", default=os.getenv("RENDER_DEPLOY_HOOK"), help="Render deploy hook URL")
    parser.add_argument("--skip-backend-check", action="store_true", help="Skip waiting for backend health")
    parser.add_argument("--skip-vercel", action="store_true", help="Skip Vercel frontend deployment")
    args = parser.parse_args()

    vc_authed, vc_user = check_auth()

    if args.hook:
        trigger_deploy_hook(args.hook)

    if not args.skip_backend_check:
        verify_render_backend(args.backend_url, timeout=120)

    vercel_url = None
    if not args.skip_vercel:
        if not vc_authed:
            print("\n[!] Vercel CLI is not authenticated.")
            print("    Run in your PowerShell terminal:  npx vercel login")
            print("    Then rerun this script.")
        else:
            vercel_url = deploy_vercel(args.backend_url)

    verify_application(args.backend_url, vercel_url)


if __name__ == "__main__":
    main()
