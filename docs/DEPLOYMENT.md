# Putting the app online

This guide takes the project from "runs on a laptop" to a public website with a forecast every
trading morning. Everything is done in a web browser. No command prompt is needed.

| Part | Where it runs | Cost |
| --- | --- | --- |
| Daily forecast and grading jobs | GitHub Actions (this repository) | Free |
| Back end (FastAPI) | Render Web Service | Free |
| Front end (Next.js website) | Vercel | Free |

How the parts connect: the scheduled jobs write small files into the `live/` folder of this
repository. The back end reads those files from GitHub. The website asks the back end.

Do the steps in this order. Allow about 30 to 45 minutes.

---

## Step 1. Ensure the default branch is ready

Scheduled jobs on GitHub only run from the default branch (`main`).

The repository must be **public** (Settings → General → scroll to *Danger Zone* → *Change visibility*),
because the back end reads the `live/` files from its public address.

## Step 2. Switch on the Morning Call

1. In the repository, click **Settings** → **Actions** → **General**.
2. Under **Workflow permissions**, choose **Read and write permissions**. Click **Save**.
   (The job needs this to save each day's forecast into the repository.)
3. Click the **Actions** tab. If GitHub shows a button asking to enable workflows, click it.
4. In the left list, click **Morning Call** → **Run workflow** → leave *forecast* selected → **Run workflow**.
5. Wait about one minute and click the run to see its log.
   - On a trading day it prints the 10 calls and commits `live/latest_forecast.json`.
   - On a weekend or NSE holiday it prints "market closed". That is correct.

From now on it runs by itself every weekday: forecast at about 7:45 am IST (with a second attempt at
8:15), grading at about 9:50 am and 4:35 pm. GitHub can start scheduled jobs up to half an hour late.

> A forecast only counts in the live track record if it was made before 9:15 am IST. A run started
> by hand later in the day is shown on the site but marked "not counted".

## Step 3. Create the back end on Render

1. Create a free account at **render.com** and sign in with GitHub.
2. In the Render Dashboard, click **New +** → **Blueprint**.
3. Select this repository (`vedantdesai0765/Stock-Market-Predictor`).
4. Render automatically reads `render.yaml` and configures:
   - **Service Name:** `stock-predictor-api`
   - **Runtime:** Docker (`backend/Dockerfile` with context `.`)
   - **Plan:** Free (0.1 CPU, 512 MB RAM)
   - **Health Check Path:** `/api/v1/health`
   - **Environment Variables:**
     - `ENVIRONMENT` = `production`
     - `MORNING_DATA_URL` = `https://raw.githubusercontent.com/vedantdesai0765/Stock-Market-Predictor/main/live`
     - `CORS_ORIGINS` = `http://localhost:3000,http://127.0.0.1:3000` (updated in Step 4)
5. Click **Apply**.
6. Render builds the Docker container and deploys the service (typically 3–5 minutes).
7. Check it works: Your API address will be `https://<service-name>.onrender.com`.
   Open `https://<service-name>.onrender.com/api/v1/health` in a browser. It should return:
   `{"status": "healthy", ...}`.

*(Optional)* To trigger deploys from GitHub Actions, copy the **Deploy Hook** URL from the Render service
settings (**Settings** → **Deploy Hook**) and add it to GitHub Secrets as `RENDER_DEPLOY_HOOK`.

## Step 4. Create the website on Vercel

1. Go to **vercel.com** and sign in with **Continue with GitHub**.
2. **Add New…** → **Project** → find this repository → **Import**.
3. **Root Directory:** click **Edit** and choose `frontend`. Vercel detects Next.js by itself.
4. Open **Environment Variables** and add:
   - Name: `NEXT_PUBLIC_API_URL`
   - Value: `https://<service-name>.onrender.com/api/v1`
5. Click **Deploy**. After about two minutes you get a live address such as `https://stock-market-predictor.vercel.app`.
6. Tell the back end to accept requests from the website:
   In the Render Dashboard → your service → **Environment** tab:
   Update `CORS_ORIGINS` to include your Vercel address:
   `https://stock-market-predictor.vercel.app,http://localhost:3000` (no trailing slash).
   Click **Save Changes**. Render automatically redeploys.
7. Open the Vercel address. The Morning Call and overview pages should load live data.

If you change `NEXT_PUBLIC_API_URL` later, open the project on Vercel → **Deployments** → **⋯** → **Redeploy**,
because that value is baked in when the Next.js site is built.

## Step 5. Keep the back end awake (recommended)

Render's free tier spins down after 15 minutes of inactivity and takes 30–50 seconds to wake up on the next request.

On GitHub: **Settings** → **Secrets and variables** → **Actions** → **Variables** → **New repository variable**.
- Name: `API_URL`
- Value: `https://<service-name>.onrender.com/api/v1`

The scheduled Morning Call job visits this URL before market hours every weekday, keeping the service awake.

## Step 6. Keep paper trades after a restart (optional)

Render's free filesystem is ephemeral (it resets whenever the service restarts, redeploys, or spins down).
That is fine for a demonstration, because the virtual wallet has a reset button in Settings anyway.

To keep trades permanently at zero cost:
1. Create a free serverless database at **neon.tech** (perpetual free tier, 0.5 GiB, never expires).
2. Copy the connection string and change the prefix from `postgresql://` to `postgresql+psycopg2://`.
3. In the Render Dashboard → your service → **Environment**, add:
   - Name: `DATABASE_URL`
   - Value: `postgresql+psycopg2://<user>:<password>@<ep-name>.neon.tech/neondb?sslmode=require`
4. Save changes. The backend connects to Neon PostgreSQL automatically.

*(Note: Render also offers a PostgreSQL add-on, but Render's free database expires and is deleted after 30 days unless upgraded to a paid $7/mo plan. Neon.tech does not expire.)*

---

## Before a presentation

- Open the website 2 minutes early so Render wakes up from sleep.
- GitHub → **Actions** → **Morning Call**: check that the morning's run is green.
- Keep the screen recording of the app at hand in case the internet fails.

## If something goes wrong

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Site takes 30-50s to load on first click | Render free tier is waking up from sleep | Wait for cold start to complete; subsequent requests are instant |
| Website loads but cards show connection error | `CORS_ORIGINS` does not match Vercel URL exactly | Check Render Environment → `CORS_ORIGINS` (no trailing slash) |
| "No forecast was published this morning" | Market holiday or Yahoo Finance download issue | Actions → Morning Call → Run workflow |
| Morning Call run fails at "Save the results" | Workflow permissions are read-only | Step 2, point 2 |
| Backend build fails on Render | Docker build context or file issue | Verify that `render.yaml` points to `dockerfilePath: backend/Dockerfile` and `dockerContext: .` |
| The site shows an old forecast | GitHub raw cache can lag by about 5 minutes | Wait and refresh |

Each December, add next year's NSE holidays to `live/nse_holidays.json`. If a year is missing, the job treats
every weekday as a trading day, and a forecast for a day on which the market did not open is marked void.
