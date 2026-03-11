# RT-DETR Crack Product (2-stage, FE + API)

Deployment-focused project for crack detection with:

1. Stage 1 (binary): crack / no crack
2. Stage 2 (5 classes): detailed crack type analysis

## Repository structure

- `backend/app.py`: FastAPI inference service
- `frontend/`: static UI served by backend
- `configs/dataset1class.yaml`: stage1 class mapping
- `configs/dataset5class.yaml`: stage2 class mapping
- `demo/app_streamlit.py`: optional Streamlit fallback

## Important note about model weights

Model files are **not committed** to GitHub (size > 100MB):

- `models/best1class.pt`
- `models/best5class.pt`

Place these files manually into `models/` before running.

## Quick start (Windows - recommended)

```bat
cd /d D:\NCKH_25-26\rtdetr_rddsplit_demo

if not exist .venv\Scripts\python.exe py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000`

## Quick start (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Health check

Cross-platform check:

```bash
python -m py_compile backend/app.py demo/app_streamlit.py
```

If you use `make` (Linux/macOS):

```bash
make check-project
```

## Main APIs

- `GET /api/health`
- `POST /api/analyze/basic`
- `POST /api/analyze/deep`
- `POST /api/stream/session/start`
- `POST /api/stream/frame/basic`
- `GET /api/stream/session/{session_id}/summary`
- `GET /api/stream/session/{session_id}/report`
- `GET /api/stream/session/{session_id}/report/bundle`
- `POST /api/stream/session/{session_id}/reset`
- `DELETE /api/stream/session/{session_id}`
- `GET /api/audit/events`

Analyze endpoints (`basic`, `deep`) accept form-data:

- `file`
- `conf` (default: basic `0.25`, deep `0.18`)
- `iou` (default `0.6`)
- `imgsz` (default: basic `640`, deep `896`)
- `device` (`auto` by default)
- `input_source` (`upload|camera|stream`)
- `scene` (`auto|default|near|far|night`)

## Current inference behavior

- Stage1:
  - triage output: `positive/review/negative`
  - optional deep-assist fallback with timeout
- Stage2:
  - 5-class + optional 1-class fallback fusion (`crack_unclassified`)
  - optional TTA + tiling
  - post-filters (ROI/watermark/small-box)
  - thin-crack rescue for long thin detections
  - severity score + QA flags + class KPIs
- Stream mode:
  - realtime basic scan
  - frame stride + candidate segment summary
  - lightweight track IDs on overlay
  - report JSON / ZIP bundle export
  - audit logs persisted to sqlite

## Runtime configuration

Use `.env.example` as baseline and override by environment variables.

Key groups:

- model/data paths: `MODEL_BASIC_PATH`, `MODEL_DEEP_PATH`, `DATASET1_YAML_PATH`, `DATASET5_YAML_PATH`
- basic/deep thresholds: `BASIC_*`, `DEEP_*`
- deep robustness: `DEEP_ENABLE_TTA`, `DEEP_ENABLE_TILING`, `DEEP_MIN_AREA_*`, `DEEP_THIN_CRACK_*`
- stream: `STREAM_*`
- audit: `AUDIT_LOG_ENABLE`, `AUDIT_DB_MAX_EVENTS`
- preprocess: `PREPROCESS_*`

## Typical stream workflow

1. `POST /api/stream/session/start`
2. Send frames to `POST /api/stream/frame/basic`
3. Read segment summary from `GET /api/stream/session/{session_id}/summary`
4. Optional deep scan on selected frames (`POST /api/analyze/deep`)
5. Export product report (`/report` or `/report/bundle`)

