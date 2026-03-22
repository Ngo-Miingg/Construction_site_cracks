# RT-DETR Crack Product (5-class, FE + API)

Deployment-focused project for crack detection using a single 5-class RT-DETR model.

## Repository structure

- `backend/app.py`: FastAPI inference service
- `frontend/`: static UI served by backend
- `configs/dataset5class.yaml`: stage2 class mapping
- `demo/app_streamlit.py`: optional Streamlit fallback

## Model weights (Git LFS)

Model files are committed via **Git LFS**:

- `models/best5class.pt`

Current runtime uses `models/best5class.pt` as the main 5-class model.

After clone, run:

```bash
git lfs install
git lfs pull
```

## Quick start (Windows - recommended)

```bat
cd /d D:\NCKH_25-26\rtdetr_rddsplit_demo
git lfs install
git lfs pull

if not exist .venv\Scripts\python.exe py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000`

## Quick start (Linux/macOS)

```bash
git lfs install
git lfs pull
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
- `GET /api/projects`
- `POST /api/projects`
- `GET /api/history`
- `DELETE /api/history`
- `GET /api/history/{history_id}/artifact/{kind}`
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
- `conf` (default `0.25`)
- `iou` (default `0.6`)
- `imgsz` (default `640`)
- `device` (`auto` by default)
- `input_source` (`upload|camera|stream`)
- `scene` (`auto|default|near|far|night`)

## Current inference behavior

- 5-class detection (single model) with class-wise boxes
- optional TTA + tiling
- post-filters (ROI/watermark/small-box)
- thin-crack rescue for long thin detections
- severity score + QA flags + class KPIs
- history persistence in sqlite (`/api/history`)
- project-based organization (create/select project, save input/output artifacts per project)
- Stream mode:
  - realtime basic scan
  - frame stride + candidate segment summary
  - lightweight track IDs on overlay
  - report JSON / ZIP bundle export
  - audit logs persisted to sqlite

## Runtime configuration

Use `.env.example` as baseline and override by environment variables.

Key groups:

- model/data paths: `MODEL_5CLASS_PATH`, `MODEL_DEEP_PATH`, `DATASET5_YAML_PATH`
- storage paths: `STREAM_STATE_DB_PATH`, `PROJECT_STORAGE_ROOT`
- threshold/runtime: `BASIC_*`, `DEEP_*`
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
