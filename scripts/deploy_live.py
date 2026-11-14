"""
scripts/deploy_live.py
Automated end-to-end deployment script for Hugging Face Spaces (Backend) and Vercel (Frontend).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "hf_space"


def check_auth():
    print("[1/6] Checking authentication status...")
    hf_authed = False
    hf_username = None
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        hf_username = user_info.get("name")
        hf_authed = bool(hf_username)
        print(f"  [+] Hugging Face: Authenticated as '{hf_username}'")
    except Exception as exc:
        print("  [-] Hugging Face: Not authenticated")

    vc_authed = False
    try:
        res = subprocess.run(["npx", "--yes", "vercel", "whoami"],
                             cwd=str(ROOT), capture_output=True, text=True, shell=True)
        out = (res.stdout + res.stderr).strip()
        if res.returncode == 0 and "Logged out" not in out:
            vc_user = [line for line in out.splitlines() if line and not line.startswith("Vercel")][0]
            vc_authed = True
            print(f"  [+] Vercel: Authenticated as '{vc_user}'")
        else:
            print("  [-] Vercel: Not authenticated (Logged out)")
    except Exception as exc:
        print(f"  [-] Vercel check failed: {exc}")

    return hf_authed, hf_username, vc_authed


def prepare_hf_payload():
    print("[2/6] Preparing backend deployment package in build/hf_space...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Core deployment manifests
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", BUILD_DIR / "README.md")
    shutil.copy2(ROOT / "backend" / "Dockerfile", BUILD_DIR / "Dockerfile")
    shutil.copy2(ROOT / ".dockerignore", BUILD_DIR / ".dockerignore")

    # Code and runtime directories
    for folder in ["backend", "src", "models", "data", "reports", "sentiment_study", "live"]:
        src_path = ROOT / folder
        dst_path = BUILD_DIR / folder
        if src_path.exists():
            shutil.copytree(src_path, dst_path)

    # Remove non-API study data and large parquet files to stay well within limits
    for rm_dir in [
        BUILD_DIR / "sentiment_study" / "data",
        BUILD_DIR / "sentiment_study" / "src",
        BUILD_DIR / "reports" / "screenshots",
    ]:
        if rm_dir.exists():
            shutil.rmtree(rm_dir, ignore_errors=True)

    for p in (BUILD_DIR / "sentiment_study" / "reports").glob("*.parquet"):
        try:
            p.unlink()
        except OSError:
            pass

    app_db = BUILD_DIR / "data" / "app.db"
    if app_db.exists():
        app_db.unlink()

    # Clean __pycache__ and .pyc
    for pyc in BUILD_DIR.rglob("*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass
    for cache in BUILD_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(cache, ignore_errors=True)
        except OSError:
            pass

    size_mb = sum(f.stat().st_size for f in BUILD_DIR.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  [+] Payload staged: {size_mb:.1f} MB total (clean & optimized for Docker build)")


def deploy_huggingface(hf_username):
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{hf_username}/stock-predictor-api"
    print(f"[3/6] Configuring and deploying Hugging Face Space: {repo_id}...")

    # 1. Create or ensure Space exists
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print("  [+] Space repository confirmed on Hugging Face.")

    # 2. Configure environment variables
    api.add_space_variable(repo_id=repo_id, key="ENVIRONMENT", value="production")
    api.add_space_variable(
        repo_id=repo_id,
        key="MORNING_DATA_URL",
        value=f"https://raw.githubusercontent.com/vedantdesai0765/Stock-Market-Predictor/main/live",
    )
    print("  [+] Space variables configured: ENVIRONMENT=production, MORNING_DATA_URL set.")

    # 3. Upload deployment payload
    print("  Uploading backend code and models to Hugging Face...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(BUILD_DIR),
        repo_type="space",
        commit_message="Deploy Stock Market Predictor backend",
    )
    print("  [+] Backend package uploaded successfully!")

    # 4. Wait for space to build and run
    print("  Waiting for Space to build and reach RUNNING status (this typically takes 3-6 minutes)...")
    direct_url = f"https://{hf_username}-stock-predictor-api.hf.space"
    start_time = time.time()
    while time.time() - start_time < 900:  # 15 minutes timeout
        try:
            runtime = api.get_space_runtime(repo_id=repo_id)
            stage = getattr(runtime, "stage", "UNKNOWN")
            print(f"  -> Space status: {stage} (elapsed: {int(time.time() - start_time)}s)")
            if stage in ("RUNNING", "APP_STARTING"):
                time.sleep(10)
                # Verify health endpoint
                try:
                    req = urllib.request.Request(f"{direct_url}/api/v1/health", headers={"User-Agent": "AntigravityDeploy"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "healthy":
                            print(f"  [+] Backend is LIVE and healthy at: {direct_url}")
                            return direct_url
                except Exception:
                    pass
            elif "ERROR" in stage.upper() or "FAILED" in stage.upper():
                print(f"  [-] Space build failed with status: {stage}")
                break
        except Exception as exc:
            print(f"  Polling notice: {exc}")
        time.sleep(15)

    return direct_url


def deploy_vercel(backend_url):
    print("[4/6] Deploying Next.js frontend to Vercel...")
    api_url = f"{backend_url}/api/v1"
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
    print("  Vercel output:")
    for line in out.splitlines():
        print(f"    {line}")

    # Extract production URL
    vercel_url = None
    for line in out.splitlines():
        clean = line.strip()
        if clean.startswith("https://") and "vercel.app" in clean:
            vercel_url = clean
            break

    if not vercel_url:
        # Fallback to inspecting vercel project
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
        print("  [-] Warning: Could not automatically parse Vercel production URL from output.")

    return vercel_url


def link_cors(hf_username, vercel_url):
    if not vercel_url:
        return
    print(f"[5/6] Linking CORS on Hugging Face to Vercel domain: {vercel_url}...")
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = f"{hf_username}/stock-predictor-api"
    api.add_space_variable(repo_id=repo_id, key="CORS_ORIGINS", value=vercel_url)
    print("  [+] CORS_ORIGINS updated on Hugging Face Space.")


def verify_application(backend_url, vercel_url):
    print("[6/6] Verifying end-to-end communication...")
    # Test Backend Health
    health_ok = False
    try:
        req = urllib.request.Request(f"{backend_url}/api/v1/health", headers={"User-Agent": "AntigravityDeploy"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            health_ok = data.get("status") == "healthy"
            print(f"  [+] Backend /health: {data}")
    except Exception as exc:
        print(f"  [-] Backend /health check failed: {exc}")

    # Test Morning Call endpoint
    try:
        req = urllib.request.Request(f"{backend_url}/api/v1/morning-call", headers={"User-Agent": "AntigravityDeploy"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [+] Backend /morning-call status: {data.get('status')}")
    except Exception as exc:
        print(f"  [-] Backend /morning-call check failed: {exc}")

    # Test Vercel Frontend
    if vercel_url:
        try:
            req = urllib.request.Request(vercel_url, headers={"User-Agent": "AntigravityDeploy"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  [+] Vercel homepage HTTP status: {resp.status}")
        except Exception as exc:
            print(f"  [-] Vercel homepage check failed: {exc}")

    print("\n================ DEPLOYMENT SUMMARY ================")
    print(f"Backend API URL:    {backend_url}")
    print(f"Interactive Docs:   {backend_url}/docs")
    print(f"Health Check:       {backend_url}/api/v1/health")
    if vercel_url:
        print(f"Public Website URL: {vercel_url}")
    print("====================================================")


def main():
    hf_authed, hf_username, vc_authed = check_auth()
    if not hf_authed or not vc_authed:
        print("\n[!] MISSING CLI AUTHENTICATION:")
        if not hf_authed:
            print("  * Hugging Face CLI is not authenticated.")
            print("    Run in your PowerShell terminal:  .venv\\Scripts\\huggingface-cli.exe login")
        if not vc_authed:
            print("  * Vercel CLI is not authenticated.")
            print("    Run in your PowerShell terminal:  npx vercel login")
        print("\nAfter logging in to both CLIs in your terminal, rerun this script to complete the deployment.")
        sys.exit(1)

    prepare_hf_payload()
    backend_url = deploy_huggingface(hf_username)
    vercel_url = deploy_vercel(backend_url)
    link_cors(hf_username, vercel_url)
    verify_application(backend_url, vercel_url)


if __name__ == "__main__":
    main()
