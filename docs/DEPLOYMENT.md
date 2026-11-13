# Putting the app online

This guide takes the project from "runs on a laptop" to a public website with a forecast every
trading morning. Everything is done in a web browser. No command prompt is needed.

| Part | Where it runs | Cost |
| --- | --- | --- |
| Daily forecast and grading jobs | GitHub Actions (this repository) | Free |
| Back end (FastAPI) | Hugging Face Spaces | Free |
| Front end (Next.js website) | Vercel | Free |

How the parts connect: the scheduled jobs write small files into the `live/` folder of this
repository. The back end reads those files from GitHub. The website asks the back end.

Do the steps in this order. Allow about one hour.

---

## Step 1. Merge the branch into `main`

Scheduled jobs on GitHub only run from the default branch (`main`).

1. Open the repository on github.com.
2. Click **Pull requests** → **New pull request**.
3. Set **base** to `main` and **compare** to `feature/morning-call-app`. Click **Create pull request**, then **Merge pull request**.

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

## Step 3. Create the back end on Hugging Face

1. Create a free account at **huggingface.co**.
2. Click your picture (top right) → **New Space**.
   - **Space name:** `stock-predictor-api`
   - **SDK:** Docker → **Blank**
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
   - Click **Create Space**.
3. Make an access token: your picture → **Access Tokens** → **Create new token** → type **Write** →
   name it `github-deploy` → **Create token**. Copy the token (it starts with `hf_`).
4. Back on GitHub: **Settings** → **Secrets and variables** → **Actions**.
   - Tab **Secrets** → **New repository secret**. Name `HF_TOKEN`, value = the token. **Add secret**.
   - Tab **Variables** → **New repository variable**. Name `HF_SPACE`,
     value = `your-huggingface-username/stock-predictor-api`. **Add variable**.
5. GitHub **Actions** tab → **Deploy back end** → **Run workflow** → **Run workflow**.
   When it turns green, open the Space on Hugging Face. It builds for 5 to 10 minutes, then shows **Running**.
6. In the Space: **Settings** → **Variables and secrets** → **New variable** (add each of these):

   | Name | Value |
   | --- | --- |
   | `ENVIRONMENT` | `production` |
   | `MORNING_DATA_URL` | `https://raw.githubusercontent.com/<github-owner>/<repository>/main/live` |
   | `CORS_ORIGINS` | leave for Step 4 |

7. Check it works. Your API address is `https://<username>-stock-predictor-api.hf.space`.
   Open `https://<username>-stock-predictor-api.hf.space/api/v1/health` in a browser. It should say `"status": "healthy"`.

## Step 4. Create the website on Vercel

1. Go to **vercel.com** and sign in with **Continue with GitHub**.
2. **Add New…** → **Project** → find this repository → **Import**.
3. **Root Directory:** click **Edit** and choose `frontend`. Vercel detects Next.js by itself.
4. Open **Environment Variables** and add:
   - Name `NEXT_PUBLIC_API_URL`
   - Value `https://<username>-stock-predictor-api.hf.space/api/v1`
5. Click **Deploy**. After about two minutes you get an address such as `https://stock-market-predictor.vercel.app`.
   (To change the name: Project → **Settings** → **Domains**.)
6. Tell the back end to accept requests from the website. In the Hugging Face Space →
   **Settings** → **Variables and secrets**, add the variable `CORS_ORIGINS` with your Vercel address,
   for example `https://stock-market-predictor.vercel.app` (no slash at the end). The Space restarts by itself.
7. Open the Vercel address. The Morning Call page should load.

If you change `NEXT_PUBLIC_API_URL` later, open the project on Vercel → **Deployments** → **⋯** → **Redeploy**,
because that value is fixed when the site is built.

## Step 5. Keep the back end awake (optional, recommended)

A free Space goes to sleep after 48 hours without visitors and takes about a minute to wake up.

On GitHub: **Settings** → **Secrets and variables** → **Actions** → **Variables** → **New repository variable**.
Name `API_URL`, value `https://<username>-stock-predictor-api.hf.space/api/v1`.
The Morning Call job then visits the API every time it runs.

## Step 6. Keep paper trades after a restart (optional)

The virtual wallet is stored in a small database file that is wiped whenever the Space restarts.
That is fine for a demonstration, because the Settings page has a reset button anyway.

To keep trades permanently, create a free database at **neon.tech**, copy its connection string, change the
beginning from `postgresql://` to `postgresql+psycopg2://`, and add it to the Space as a **secret** named
`DATABASE_URL`.

---

## Before a presentation

- Open the website 5 minutes early, so the back end is awake.
- GitHub → **Actions** → **Morning Call**: check that the morning's run is green.
- Keep the screen recording of the app at hand in case the internet fails.

## If something goes wrong

| What you see | Likely cause | What to do |
| --- | --- | --- |
| "The Morning Call could not be loaded" | The Space is asleep or rebuilding | Wait a minute and press Retry |
| Website loads but every page shows an error | `CORS_ORIGINS` does not match the Vercel address exactly | Fix the variable in the Space (no trailing slash) |
| "No forecast was published this morning" | Yahoo Finance refused the download | Actions → Morning Call → Run workflow. If it is after 9:15 am the day is not counted |
| Morning Call run fails at "Save the results" | Workflow permissions are read-only | Step 2, point 2 |
| Deploy back end fails at "Check the setup" | `HF_TOKEN` or `HF_SPACE` is missing | Step 3, point 4 |
| The site shows an old forecast | GitHub's public file address can lag by about 5 minutes | Wait and refresh |
| No scheduled runs at all | Workflows are disabled, or the branch is not merged into `main` | Steps 1 and 2 |

Each December, add next year's NSE holidays to `live/nse_holidays.json`. If a year is missing, the job treats
every weekday as a trading day, and a forecast for a day on which the market did not open is marked void.
