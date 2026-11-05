# Production Deployment Guide: Stock Market Predictor

**Platform:** Stock Market Predictor — Institutional Financial Analytics Terminal
**Document:** Deployment & Operations Guide
**Version:** 1.0.0 (Phase 6 Final Release)
**Date:** November 2026

---

## 1. Overview & Architecture

The Stock Market Predictor terminal is composed of two decoupled services:
1. **Backend Service:** FastAPI Python application running on Uvicorn ASGI (`port 8000`). Exposes machine learning pipelines, walk-forward forecasting models, news sentiment explainability, and paper trading database operations.
2. **Frontend Service:** Next.js 14 application with React 19, TypeScript, Tailwind CSS, shadcn/ui, and Recharts (`port 3000`). Connects to FastAPI via REST API endpoints.
3. **Optional Legacy Dashboard:** Streamlit research application (`port 8501`), maintained for backward compatibility.

```
+----------------------------------------------------------------------------------------------------+
|                                    DEPLOYMENT ARCHITECTURE                                         |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    Browser Clients (Desktop / Mobile)                                                              |
|          |                                                                                         |
|          v                                                                                         |
|    +-----------------------------+                                                                 |
|    | Reverse Proxy (Nginx / ALB) |                                                                 |
|    +--------------+--------------+                                                                 |
|                   |                                                                                |
|         +---------+---------+                                                                      |
|         |                   |                                                                      |
|         v                   v                                                                      |
|   +-----------+       +-----------+                                                                |
|   |  Next.js  |       |  FastAPI  | <-----> [ SQLite / PostgreSQL Database ]                       |
|   | (Port 3000)       | (Port 8000)                                                                |
|   +-----------+       +-----+-----+                                                                |
|                             |                                                                      |
|                             v                                                                      |
|                     [ Models & Data ]                                                              |
|                     - 10 Stock OHLCV Datasets (2018-2026)                                          |
|                     - Pre-trained Classifier Artifacts                                             |
|                     - FinBERT Company Sentiment Scores                                             |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Environment Variables Configuration

Copy `.env.example` to `.env` in the project root:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

### Configuration Variables Reference

| Variable | Description | Default (Local Dev) | Production Example |
|---|---|---|---|
| `ENVIRONMENT` | Runtime environment name | `development` | `production` |
| `PORT` | Backend listen port | `8000` | `8000` |
| `CORS_ORIGINS` | Whitelisted frontend origins | `http://localhost:3000,http://127.0.0.1:3000` | `https://terminal.yourdomain.com` |
| `DATABASE_URL` | SQLAlchemy database connection URI | `sqlite:///data/app.db` | `postgresql://usr:pwd@db:5432/stock_market` |
| `NEWSAPI_KEY` | Optional developer key from newsapi.org | `your_newsapi_key_here` | `a1b2c3d4e5f6...` |
| `NEXT_PUBLIC_API_URL` | Frontend client API base URL | `http://127.0.0.1:8000/api/v1` | `https://api.yourdomain.com/api/v1` |

---

## 3. Local Bare-Metal Execution (Windows / Linux)

### 3.1 Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ (Tested on Node.js 20)
- npm 9+

### 3.2 Backend Setup & Execution
```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Launch FastAPI backend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Verify API health: `http://127.0.0.1:8000/api/v1/health`
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`

### 3.3 Frontend Setup & Execution
```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm ci

# 3. Build optimized production bundle
npm run build

# 4. Start production server
npm run start
```
- Web Application: `http://localhost:3000`

### 3.4 (Optional) Run Streamlit Research Fallback
```powershell
# From repository root
streamlit run app.py --server.port 8501
```

---

## 4. Docker & Multi-Container Deployment

The repository includes a ready-to-run `docker-compose.yml` defining the backend and frontend microservices.

### 4.1 Launch with Docker Compose
```bash
# Build and run containers in background
docker compose up -d --build

# Verify container status
docker compose ps

# View live application logs
docker compose logs -f
```

### 4.2 Health Checks
The backend container includes an automated healthcheck testing `/api/v1/health` every 30 seconds:
```bash
docker inspect --format='{{json .State.Health.Status}}' stock-market-backend
# Output: "healthy"
```

### 4.3 Stopping Containers
```bash
docker compose down
```

---

## 5. Production Database Migration (SQLite to PostgreSQL)

For single-server deployments, the default SQLite database (`data/app.db`) provides optimal performance, zero maintenance, and full transactional ACID compliance.

To scale horizontally across multiple container instances, switch to PostgreSQL:
1. Provision a PostgreSQL instance (v14+).
2. Update `DATABASE_URL` in your `.env` or container environment:
   ```bash
   DATABASE_URL=postgresql://stock_admin:StrongPassword@postgres-host:5432/stock_market_db
   ```
3. Install PostgreSQL driver if needed:
   ```bash
   pip install psycopg2-binary
   ```
4. Restart the FastAPI application. Tables (`wallets`, `holdings`, `transactions`) are automatically generated on startup via `init_db()`.

---

## 6. Production Reverse Proxy Configuration (Nginx)

Below is an institutional Nginx reverse proxy configuration with TLS 1.3, rate limiting, and HTTP/2 support:

```nginx
# Rate Limiting Zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;

server {
    listen 80;
    server_name terminal.yourdomain.com api.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name terminal.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/terminal.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/terminal.yourdomain.com/privkey.pem;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Frontend Route (Next.js)
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Backend API Route (FastAPI)
    location / {
        limit_req zone=api_limit burst=40 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 7. Verification Checklist

- [x] Backend returns `{"status": "healthy"}` on `GET /api/v1/health`.
- [x] Frontend renders properly at `http://localhost:3000`.
- [x] All 10 stocks load in the global stock selector.
- [x] Reset wallet confirmation modal executes atomic reset to ₹1,00,000.
- [x] Buy and Sell orders calculate 7.5 bps statutory friction.
- [x] Interactive Recharts render without console errors.
- [x] All 25 automated backend pytest tests pass.
