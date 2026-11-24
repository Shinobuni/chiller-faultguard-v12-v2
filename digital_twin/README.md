# Chiller FaultGuard v12.0 — Cyber-Physical Digital Twin (Vercel Web Edition)

> **SEMIEXPO Vietnam 2026 · Tech Zoomers Challenge (Pod 2 — Team 8)**  
> Read-Only Advisory Digital Twin running **Quan's Frozen HistGradientBoostingClassifier** (`quan_hgb_model.joblib`, SHA-256 `6eeda35f`, `0.00e+00` LBNL Oct-01 parity), **8-Node Causal Root-Cause DAG**, **EVN TOU Break-Even PM Optimizer**, **Air-Gapped Evidence-Grounded RCA Agent**, and **Real-World 3-Stage Vision ML (`fouling_cv_model.joblib`, ArUco `DICT_4X4_50` + 1-Tap Calibration) + Live 16 kHz PCM Audio FFT**.

---

## Live Routes

| Route | Description |
| :--- | :--- |
| `/` | Main 5-Tab Minimal ISA-101 Digital Twin Dashboard |
| `/mobile-cam` | iPhone 11 Landscape Optical & Mic FFT Companion (HTTPS Live Camera + Mic natively supported on Vercel!) |
| `/hardware-3d-viewer` | Interactive 15-Part Mini-CT1 v12 3D CAD Assembly Viewer (100% self-contained) |
| `/api/health` | Service health & model integrity SHA-256 check (`version: "12.0.0"`) |
| `/api/hardware/aruco-sheet?format=png` | Printable A4 300-DPI ArUco 4-Corner Calibration Sheet (PNG) |
| `/api/hardware/aruco-sheet?format=pdf` | Printable A4 300-DPI ArUco 4-Corner Calibration Sheet (PDF) |

---

## Deploy to Vercel (`vercel.app`)

This folder (`digital_twin/`) is pre-configured for **Vercel Native FastAPI (Python 3.14)**:
- Root ASGI entrypoint: [`main.py`](main.py) (`app = FastAPI(...)`)
- Python runtime & pinned ML wheels: [`pyproject.toml`](pyproject.toml), [`requirements.txt`](requirements.txt), [`.python-version`](.python-version) (`3.14`)
- Vercel function config: [`vercel.json`](vercel.json) (`regions: ["sin1"]` Singapore for low Vietnam/APAC latency, `maxDuration: 30`)
- Performance optimized: `GZipMiddleware` + static asset caching, cached HGB scenario scores (`0.00e+00` parity), memoized synthetic Vision frames, lightweight JPEG Tab 5 previews, non-blocking parallel boot & active-tab-only DOM/SVG rendering
- Bundle hygiene: [`.vercelignore`](.vercelignore) excludes tests & dev caches
- Read-only filesystem safe: live 1-Tap Bench Calibration automatically writes to `/tmp` when running on Vercel
- Vercel 4.5 MB payload safe: uploaded camera photos > 1.8 MB are automatically downscaled client-side to ≤ 1600px JPEG before inspection

### Option 1 — GitHub + Vercel Dashboard (Zero CLI needed)
1. Create a new GitHub repository and upload all files inside `digital_twin/`.
2. Go to **vercel.com** → **Add New… → Project** → Import your GitHub repo.
3. Click **Deploy** (Vercel auto-detects FastAPI + Python 3.14 from `main.py` & `pyproject.toml`).

### Option 2 — Vercel CLI
```bash
cd digital_twin
npx vercel --prod
```

---

## Run Locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://127.0.0.1:8000`.
