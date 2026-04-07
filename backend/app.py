from __future__ import annotations

import os
import time
import threading
import uuid
import json
import sqlite3
import io
import csv
import zipfile
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


def env_path(name: str, *defaults: Path) -> Path:
    raw = os.getenv(name)
    if raw:
        return Path(raw)
    for candidate in defaults:
        if candidate.exists():
            return candidate
    return defaults[0]


MODEL_5CLASS_PATH = env_path(
    "MODEL_5CLASS_PATH",
    ROOT / "models" / "best5class.pt",
    ROOT / "best5class.pt",
)
MODEL_DEEP_PATH = env_path("MODEL_DEEP_PATH", MODEL_5CLASS_PATH)
MODEL_BASIC_PATH = MODEL_DEEP_PATH
DATASET5_YAML_PATH = env_path(
    "DATASET5_YAML_PATH",
    ROOT / "configs" / "dataset5class.yaml",
    ROOT / "dataset5class.yaml",
)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    out: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except Exception:
            continue
    return out or list(default)


CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())
AUTO_DEVICE = "0" if CUDA_AVAILABLE else "cpu"


MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 20)
BASIC_CONF_MIN = env_float("BASIC_CONF_MIN", 0.25)
BASIC_POS_CONF = env_float("BASIC_POS_CONF", 0.35)
BASIC_REVIEW_CONF = env_float("BASIC_REVIEW_CONF", 0.20)
BASIC_IMGSZ_DEFAULT = env_int("BASIC_IMGSZ_DEFAULT", 640)
BASIC_ENABLE_DEEP_ASSIST = env_bool("BASIC_ENABLE_DEEP_ASSIST", False)
BASIC_DEEP_ASSIST_CONF = env_float("BASIC_DEEP_ASSIST_CONF", 0.25)
BASIC_DEEP_ASSIST_IMGSZ = env_int("BASIC_DEEP_ASSIST_IMGSZ", 640)
BASIC_DEEP_ASSIST_IOU = env_float("BASIC_DEEP_ASSIST_IOU", 0.60)
DEEP_CONF_DEFAULT = env_float("DEEP_CONF_DEFAULT", 0.25)
DEEP_IMGSZ_DEFAULT = env_int("DEEP_IMGSZ_DEFAULT", 640)
DEEP_FALLBACK_BASIC_CONF = env_float("DEEP_FALLBACK_BASIC_CONF", 0.25)
DEEP_FUSION_IOU = env_float("DEEP_FUSION_IOU", 0.20)
DEEP_ENABLE_BASIC_FALLBACK = env_bool("DEEP_ENABLE_BASIC_FALLBACK", False)
DEEP_ENABLE_TTA = env_bool("DEEP_ENABLE_TTA", False)
DEEP_TTA_SCALES = env_float_list("DEEP_TTA_SCALES", [1.0])
DEEP_TTA_HFLIP = env_bool("DEEP_TTA_HFLIP", False)
DEEP_ENABLE_TILING = env_bool("DEEP_ENABLE_TILING", False)
DEEP_TILE_SIZE = env_int("DEEP_TILE_SIZE", 640)
DEEP_TILE_OVERLAP = env_int("DEEP_TILE_OVERLAP", 128)
DEEP_ENSEMBLE_NMS_IOU = env_float("DEEP_ENSEMBLE_NMS_IOU", 0.70)
DEEP_MAX_RETURN = env_int("DEEP_MAX_RETURN", 300)
DEEP_ENABLE_ROI_FILTER = env_bool("DEEP_ENABLE_ROI_FILTER", False)
DEEP_ROI_TOP_EXCLUDE_RATIO = env_float("DEEP_ROI_TOP_EXCLUDE_RATIO", 0.08)
DEEP_ENABLE_WATERMARK_MASK = env_bool("DEEP_ENABLE_WATERMARK_MASK", False)
DEEP_WATERMARK_RIGHT_RATIO = env_float("DEEP_WATERMARK_RIGHT_RATIO", 0.45)
DEEP_WATERMARK_BOTTOM_RATIO = env_float("DEEP_WATERMARK_BOTTOM_RATIO", 0.24)
DEEP_MIN_AREA_PX = env_float("DEEP_MIN_AREA_PX", 16.0)
DEEP_MIN_AREA_RATIO = env_float("DEEP_MIN_AREA_RATIO", 0.000003)
DEEP_THIN_CRACK_RESCUE = env_bool("DEEP_THIN_CRACK_RESCUE", False)
DEEP_THIN_CRACK_MIN_AR = env_float("DEEP_THIN_CRACK_MIN_AR", 6.0)
DEEP_THIN_CRACK_MIN_CONF = env_float("DEEP_THIN_CRACK_MIN_CONF", 0.18)
QA_LOW_CONF_THRESH = env_float("QA_LOW_CONF_THRESH", 0.45)
QA_MIN_KEEP_RATIO = env_float("QA_MIN_KEEP_RATIO", 0.35)
USE_FP16_ON_CUDA = env_bool("USE_FP16_ON_CUDA", True)
STREAM_IMGSZ = env_int("STREAM_IMGSZ", 512)
STREAM_IOU = env_float("STREAM_IOU", 0.60)
STREAM_BASIC_CONF = env_float("STREAM_BASIC_CONF", 0.20)
STREAM_CANDIDATE_CONF = env_float("STREAM_CANDIDATE_CONF", 0.35)
STREAM_DEVICE = os.getenv("STREAM_DEVICE", "auto")
STREAM_FRAME_STRIDE = env_int("STREAM_FRAME_STRIDE", 3)
STREAM_SEGMENT_GAP_SEC = env_float("STREAM_SEGMENT_GAP_SEC", 1.2)
STREAM_TRACK_ENABLE = env_bool("STREAM_TRACK_ENABLE", True)
STREAM_TRACK_IOU = env_float("STREAM_TRACK_IOU", 0.25)
STREAM_TRACK_MAX_AGE = env_int("STREAM_TRACK_MAX_AGE", 10)
STREAM_MAX_SESSIONS = env_int("STREAM_MAX_SESSIONS", 32)
STREAM_SESSION_TTL_SEC = env_int("STREAM_SESSION_TTL_SEC", 4 * 3600)
STREAM_STATE_DB_PATH = Path(os.getenv("STREAM_STATE_DB_PATH", ROOT / "runs" / "stream_state.db"))
PROJECT_STORAGE_ROOT = Path(os.getenv("PROJECT_STORAGE_ROOT", ROOT / "runs" / "projects"))
STREAM_DB_SYNC_EVERY = env_int("STREAM_DB_SYNC_EVERY", 1)
AUDIT_LOG_ENABLE = env_bool("AUDIT_LOG_ENABLE", True)
AUDIT_DB_MAX_EVENTS = env_int("AUDIT_DB_MAX_EVENTS", 50000)
PREPROCESS_ENABLE = env_bool("PREPROCESS_ENABLE", False)
PREPROCESS_ENABLE_CLAHE = env_bool("PREPROCESS_ENABLE_CLAHE", False)
PREPROCESS_CLAHE_CLIP = env_float("PREPROCESS_CLAHE_CLIP", 2.0)
PREPROCESS_CLAHE_TILE = env_int("PREPROCESS_CLAHE_TILE", 8)
PREPROCESS_CAMERA_ALPHA = env_float("PREPROCESS_CAMERA_ALPHA", 1.08)
PREPROCESS_CAMERA_BETA = env_float("PREPROCESS_CAMERA_BETA", 2.0)
PREPROCESS_NIGHT_ALPHA = env_float("PREPROCESS_NIGHT_ALPHA", 1.20)
PREPROCESS_NIGHT_BETA = env_float("PREPROCESS_NIGHT_BETA", 5.0)
BASIC_DEEP_ASSIST_TIMEOUT_MS = env_int("BASIC_DEEP_ASSIST_TIMEOUT_MS", 900)
BASIC_SCENE = os.getenv("BASIC_SCENE", "default").strip().lower()
DEEP_SCENE = os.getenv("DEEP_SCENE", "default").strip().lower()
ENABLE_SCENE_PROFILE = env_bool("ENABLE_SCENE_PROFILE", False)
BASIC_ENABLE_POSTPROCESS = env_bool("BASIC_ENABLE_POSTPROCESS", False)

DEFAULT_STAGE2_NAMES = {
    0: "vet nut doc",
    1: "vet nut ngang",
    2: "nut da ca sau",
    3: "hu hong khac",
    4: "o ga",
}

CLASS_NAME_MAP_VI = {
    "crack": "vet nut",
    "longitudinal crack": "vet nut doc",
    "transverse crack": "vet nut ngang",
    "alligator crack": "nut da ca sau",
    "other corruption": "hu hong khac",
    "pothole": "o ga",
    "crack unclassified": "vet nut chua phan loai",
    "crack_unclassified": "vet nut chua phan loai",
}

_model_cache: dict[str, tuple[int, Any]] = {}
_dataset_names_cache: tuple[int, dict[int, str]] | None = None
_stream_sessions: dict[str, dict[str, Any]] = {}
_stream_lock = threading.Lock()
_stream_db_lock = threading.Lock()
_assist_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deep-assist")
_assist_state_lock = threading.Lock()
_assist_cooldown_until = 0.0
_stream_update_counter = 0

app = FastAPI(title="Road Crack Detector API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def read_names_from_yaml(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {}
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    names = content.get("names") if isinstance(content, dict) else None
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def get_stage2_names() -> dict[int, str]:
    """
    Return localized class names for runtime inference.

    Cache by file mtime to avoid re-reading YAML on every request while still
    reflecting updates if the dataset config changes.
    """
    global _dataset_names_cache
    try:
        version = int(DATASET5_YAML_PATH.stat().st_mtime_ns)
    except Exception:
        version = -1

    if _dataset_names_cache is not None and _dataset_names_cache[0] == version:
        return dict(_dataset_names_cache[1])

    names = read_names_from_yaml(DATASET5_YAML_PATH) or DEFAULT_STAGE2_NAMES
    localized = localize_name_map(names)
    _dataset_names_cache = (version, localized)
    return dict(localized)


def localize_class_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", " ")
    return CLASS_NAME_MAP_VI.get(key, str(name))


def localize_name_map(names_map: dict[int, str]) -> dict[int, str]:
    return {int(k): localize_class_name(v) for k, v in names_map.items()}


def infer_scene_from_image(image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    if mean_luma < 75.0:
        return "night"
    return "default"


def normalize_scene(scene: str, image_bgr: np.ndarray) -> str:
    raw = str(scene or "auto").strip().lower()
    if raw in {"", "auto"}:
        return infer_scene_from_image(image_bgr)
    if raw not in {"default", "near", "far", "night"}:
        return "default"
    return raw


def apply_scene_profile(stage: str, conf: float, imgsz: int, scene: str) -> tuple[float, int]:
    if not ENABLE_SCENE_PROFILE:
        return clamp_float(conf, 0.01, 0.99), clamp_int(imgsz, 320, 1536)
    # Scene-aware runtime profile keeps one API but adapts sensitivity by context.
    profile = {
        "default": {"conf_scale": 1.00, "imgsz_scale": 1.00},
        "near": {"conf_scale": 1.08, "imgsz_scale": 0.95},
        "far": {"conf_scale": 0.82, "imgsz_scale": 1.25},
        "night": {"conf_scale": 0.86, "imgsz_scale": 1.15},
    }
    p = profile.get(scene, profile["default"])
    tuned_conf = clamp_float(conf * float(p["conf_scale"]), 0.01, 0.99)
    tuned_size = clamp_int(round(imgsz * float(p["imgsz_scale"])), 320, 1536)
    tuned_size = clamp_int(int(round(tuned_size / 32.0) * 32), 320, 1536)
    # Keep deep path slightly stricter to reduce noise explosion after TTA/tiling.
    if stage == "deep":
        tuned_conf = clamp_float(tuned_conf + 0.01, 0.01, 0.99)
    return tuned_conf, tuned_size


def preprocess_image_for_inference(image_bgr: np.ndarray, source: str, scene: str) -> np.ndarray:
    if not PREPROCESS_ENABLE:
        return image_bgr
    out = image_bgr
    src = str(source or "upload").strip().lower()
    if src in {"camera", "stream"}:
        out = cv2.convertScaleAbs(
            out,
            alpha=max(0.5, float(PREPROCESS_CAMERA_ALPHA)),
            beta=float(PREPROCESS_CAMERA_BETA),
        )
    if scene == "night":
        out = cv2.convertScaleAbs(
            out,
            alpha=max(0.5, float(PREPROCESS_NIGHT_ALPHA)),
            beta=float(PREPROCESS_NIGHT_BETA),
        )
    if PREPROCESS_ENABLE_CLAHE:
        yuv = cv2.cvtColor(out, cv2.COLOR_BGR2YUV)
        clip = max(0.5, float(PREPROCESS_CLAHE_CLIP))
        tile = max(2, int(PREPROCESS_CLAHE_TILE))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return out


def normalize_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(conf):
        return 0.0
    return max(0.0, min(1.0, conf))


def load_model_cached(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    version = int(model_path.stat().st_mtime_ns)
    key = str(model_path.resolve())
    if key in _model_cache and _model_cache[key][0] == version:
        return _model_cache[key][1]
    # Use Ultralytics' generic loader so both YOLO and RT-DETR weights work safely.
    model = YOLO(str(model_path))
    _model_cache[key] = (version, model)
    return model


def decode_upload(uploaded_file: UploadFile) -> np.ndarray:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    raw = uploaded_file.file.read(max_bytes + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_MB} MB")
    np_arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def resolve_names(model: Any, fallback_names: dict[int, str]) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict) and names:
        return localize_name_map({int(k): str(v) for k, v in names.items()})
    return localize_name_map(fallback_names)


def normalize_device(device: str) -> str | None:
    raw = str(device or "").strip().lower()
    if raw in {"", "auto", "none"}:
        return AUTO_DEVICE
    if raw in {"gpu", "cuda", "cuda:0"}:
        return "0" if CUDA_AVAILABLE else "cpu"
    return raw


def run_prediction(
    model_path: Path,
    image_bgr: np.ndarray,
    fallback_names: dict[int, str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    model = load_model_cached(model_path)
    names_map = resolve_names(model, fallback_names)
    device_resolved = normalize_device(device)
    use_half = bool(
        USE_FP16_ON_CUDA
        and CUDA_AVAILABLE
        and device_resolved is not None
        and "cpu" not in device_resolved
    )
    t0 = time.perf_counter()

    results = model.predict(
        image_bgr,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device_resolved,
        half=use_half,
        verbose=False,
    )
    infer_time_ms = (time.perf_counter() - t0) * 1000.0
    r0 = results[0]

    detections: list[dict[str, Any]] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for box in r0.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names_map.get(cls_id, f"class_{cls_id}"),
                    "confidence": normalize_confidence(box.conf.item() if box.conf is not None else 0.0),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "detections": detections,
        "names": names_map,
        "infer_time_ms": round(float(infer_time_ms), 2),
        "device_used": device_resolved or "auto",
    }


def bbox_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def clamp_bbox_to_image(box: list[float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def nms_detections(
    detections: list[dict[str, Any]],
    iou_thr: float,
    max_keep: int,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for det in ordered:
        keep = True
        for prev in kept:
            if class_aware and int(det.get("class_id", -999)) != int(prev.get("class_id", -999)):
                continue
            if bbox_iou_xyxy(det["bbox_xyxy"], prev["bbox_xyxy"]) >= iou_thr:
                keep = False
                break
        if keep:
            kept.append(det)
            if len(kept) >= max_keep:
                break
    return kept


def build_tile_starts(total: int, tile_size: int, overlap: int) -> list[int]:
    tile = max(128, int(tile_size))
    ov = max(0, min(int(overlap), tile - 1))
    if total <= tile:
        return [0]
    step = max(64, tile - ov)
    starts = list(range(0, max(1, total - tile + 1), step))
    last = total - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_deep_ensemble(
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    fallback_names: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, str]]:
    h, w = image_bgr.shape[:2]
    all_dets: list[dict[str, Any]] = []
    pass_stats = {"full_passes": 0, "hflip_passes": 0, "tile_passes": 0}
    names_map: dict[int, str] = {}

    def append_pass(run_img: np.ndarray, run_imgsz: int, source: str) -> None:
        nonlocal names_map
        pred = run_prediction(MODEL_DEEP_PATH, run_img, fallback_names, conf, iou, run_imgsz, device)
        if not names_map:
            names_map = pred["names"]
        for det in pred["detections"]:
            all_dets.append({**det, "source": source})

    # Full-image passes (multi-scale TTA).
    scales = [1.0]
    if DEEP_ENABLE_TTA:
        for s in DEEP_TTA_SCALES:
            if s > 0:
                scales.append(float(s))
    unique_scales: list[float] = []
    for s in scales:
        if all(abs(s - x) > 1e-6 for x in unique_scales):
            unique_scales.append(s)

    for scale in unique_scales:
        run_imgsz = clamp_int(round(imgsz * scale), 320, 2048)
        append_pass(image_bgr, run_imgsz, f"deep_full_{run_imgsz}")
        pass_stats["full_passes"] += 1

    # Horizontal flip pass.
    if DEEP_ENABLE_TTA and DEEP_TTA_HFLIP:
        flipped = cv2.flip(image_bgr, 1)
        pred = run_prediction(MODEL_DEEP_PATH, flipped, fallback_names, conf, iou, imgsz, device)
        if not names_map:
            names_map = pred["names"]
        for det in pred["detections"]:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            mapped = [float(w) - x2, y1, float(w) - x1, y2]
            all_dets.append({**det, "bbox_xyxy": clamp_bbox_to_image(mapped, w, h), "source": "deep_hflip"})
        pass_stats["hflip_passes"] += 1

    # Tile passes for small/thin crack recall.
    tile_size = clamp_int(DEEP_TILE_SIZE, 320, 2048)
    tile_overlap = clamp_int(DEEP_TILE_OVERLAP, 0, tile_size - 1)
    if DEEP_ENABLE_TILING and (w > tile_size or h > tile_size):
        xs = build_tile_starts(w, tile_size, tile_overlap)
        ys = build_tile_starts(h, tile_size, tile_overlap)
        tile_imgsz = clamp_int(min(imgsz, tile_size), 320, 2048)
        for y0 in ys:
            for x0 in xs:
                x1 = x0
                y1 = y0
                x2 = min(w, x0 + tile_size)
                y2 = min(h, y0 + tile_size)
                tile = image_bgr[y1:y2, x1:x2]
                pred = run_prediction(MODEL_DEEP_PATH, tile, fallback_names, conf, iou, tile_imgsz, device)
                if not names_map:
                    names_map = pred["names"]
                for det in pred["detections"]:
                    bx1, by1, bx2, by2 = det["bbox_xyxy"]
                    mapped = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                    all_dets.append(
                        {
                            **det,
                            "bbox_xyxy": clamp_bbox_to_image(mapped, w, h),
                            "source": "deep_tile",
                        }
                    )
                pass_stats["tile_passes"] += 1

    fused = nms_detections(
        all_dets,
        iou_thr=clamp_float(DEEP_ENSEMBLE_NMS_IOU, 0.1, 0.95),
        max_keep=clamp_int(DEEP_MAX_RETURN, 10, 1000),
        class_aware=True,
    )
    return fused, pass_stats, names_map


def crack_class_weight(class_name: str) -> float:
    key = class_name.strip().lower()
    table = {
        "vet nut doc": 0.90,
        "vet nut ngang": 1.00,
        "nut da ca sau": 1.35,
        "hu hong khac": 1.10,
        "o ga": 1.60,
        "vet nut chua phan loai": 0.95,
        # Backward-compatible english labels
        "longitudinal crack": 0.90,
        "transverse crack": 1.00,
        "alligator crack": 1.35,
        "other corruption": 1.10,
        "pothole": 1.60,
        "crack_unclassified": 0.95,
        "crack": 0.95,
        "vet nut": 0.95,
    }
    return float(table.get(key, 1.0))


def enrich_detection_geometry(det: dict[str, Any], w: int, h: int) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    area_ratio = area / max(1.0, float(w * h))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    out = dict(det)
    out["bbox_xyxy"] = [x1, y1, x2, y2]
    out["area_px"] = float(area)
    out["area_ratio"] = float(area_ratio)
    out["center_xy"] = [float(cx), float(cy)]
    out["class_weight"] = crack_class_weight(str(det.get("class_name", "")))
    return out


def postprocess_basic_detections(
    detections: list[dict[str, Any]],
    w: int,
    h: int,
) -> list[dict[str, Any]]:
    if not BASIC_ENABLE_POSTPROCESS:
        raw_out: list[dict[str, Any]] = []
        for det in detections:
            raw_out.append(
                {
                    "class_id": int(det.get("class_id", 0)),
                    "class_name": localize_class_name(str(det.get("class_name", "vet nut"))),
                    "confidence": normalize_confidence(det.get("confidence", 0.0)),
                    "bbox_xyxy": clamp_bbox_to_image(
                        [float(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])], w, h
                    ),
                    "source": str(det.get("source", "basic")),
                }
            )
        return raw_out

    cleaned: list[dict[str, Any]] = []
    image_area = max(1.0, float(w * h))
    for det in detections:
        box = clamp_bbox_to_image([float(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])], w, h)
        x1, y1, x2, y2 = box
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 2.0 or bh < 2.0:
            continue
        area_ratio = (bw * bh) / image_area
        width_ratio = bw / max(1.0, float(w))
        height_ratio = bh / max(1.0, float(h))
        # Drop obviously bad "strip" boxes before showing/fusing them.
        if area_ratio > 0.75:
            continue
        if width_ratio > 0.97 and height_ratio > 0.12:
            continue
        cleaned.append(
            {
                "class_id": 0,
                "class_name": "vet nut",
                "confidence": normalize_confidence(det.get("confidence", 0.0)),
                "bbox_xyxy": box,
                "source": str(det.get("source", "basic")),
            }
        )

    return nms_detections(cleaned, iou_thr=0.45, max_keep=8, class_aware=False)


def apply_post_filters(
    detections: list[dict[str, Any]],
    w: int,
    h: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "removed_small": 0,
        "removed_top_roi": 0,
        "removed_watermark_zone": 0,
    }
    kept: list[dict[str, Any]] = []
    top_limit = clamp_float(DEEP_ROI_TOP_EXCLUDE_RATIO, 0.0, 0.40) * float(h)
    wm_right = clamp_float(DEEP_WATERMARK_RIGHT_RATIO, 0.05, 0.95) * float(w)
    wm_bottom = clamp_float(DEEP_WATERMARK_BOTTOM_RATIO, 0.05, 0.95) * float(h)
    min_area_px = max(1.0, float(DEEP_MIN_AREA_PX))
    min_area_ratio = clamp_float(DEEP_MIN_AREA_RATIO, 0.0, 0.01)
    thin_ar_min = max(1.0, float(DEEP_THIN_CRACK_MIN_AR))
    thin_conf_min = clamp_float(DEEP_THIN_CRACK_MIN_CONF, 0.01, 0.99)

    for det in detections:
        item = enrich_detection_geometry(det, w, h)
        cx, cy = item["center_xy"]
        x1, y1, x2, y2 = item["bbox_xyxy"]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
        cls = str(item.get("class_name", "")).strip().lower().replace("_", " ")
        linear_crack_cls = cls in {
            "vet nut",
            "vet nut doc",
            "vet nut ngang",
            "vet nut chua phan loai",
            "crack",
            "longitudinal crack",
            "transverse crack",
            "crack unclassified",
            "crack unclassified",
        }
        thin_crack_rescue = (
            bool(DEEP_THIN_CRACK_RESCUE)
            and linear_crack_cls
            and aspect_ratio >= thin_ar_min
            and float(item.get("confidence", 0.0)) >= thin_conf_min
        )

        if (item["area_px"] < min_area_px or item["area_ratio"] < min_area_ratio) and not thin_crack_rescue:
            stats["removed_small"] += 1
            continue
        if DEEP_ENABLE_ROI_FILTER and cy < top_limit:
            stats["removed_top_roi"] += 1
            continue
        if DEEP_ENABLE_WATERMARK_MASK and cx < wm_right and cy < wm_bottom:
            stats["removed_watermark_zone"] += 1
            continue
        kept.append(item)

    return kept, stats


def summarize_by_class(
    detections: list[dict[str, Any]],
    image_area: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for det in detections:
        name = str(det.get("class_name", "unknown"))
        cls_id = int(det.get("class_id", -1))
        info = grouped.setdefault(
            name,
            {
                "class_id": cls_id,
                "class_name": name,
                "count": 0,
                "mean_conf": 0.0,
                "max_conf": 0.0,
                "total_area_px": 0.0,
                "total_area_ratio": 0.0,
                "weighted_count": 0.0,
            },
        )
        conf = float(det.get("confidence", 0.0))
        area_px = float(det.get("area_px", 0.0))
        info["count"] += 1
        info["mean_conf"] += conf
        info["max_conf"] = max(info["max_conf"], conf)
        info["total_area_px"] += area_px
        info["weighted_count"] += float(det.get("class_weight", 1.0))

    out: list[dict[str, Any]] = []
    for v in grouped.values():
        count = max(1, int(v["count"]))
        v["mean_conf"] = float(v["mean_conf"] / count)
        v["total_area_ratio"] = float(v["total_area_px"] / max(1.0, image_area))
        out.append(v)

    out.sort(key=lambda x: (x["count"], x["total_area_ratio"]), reverse=True)
    return out


def compute_severity(
    detections: list[dict[str, Any]],
    w: int,
    h: int,
) -> dict[str, Any]:
    image_area = max(1.0, float(w * h))
    if not detections:
        return {
            "severity_score": 0.0,
            "severity_level": "A",
            "recommendation": "Mat duong hien tai on dinh, tiep tuc giam sat dinh ky.",
            "metrics": {
                "count": 0,
                "mean_conf": 0.0,
                "total_area_ratio": 0.0,
                "max_area_ratio": 0.0,
                "weighted_count": 0.0,
                "weighted_area_ratio": 0.0,
                "density_per_mpx": 0.0,
            },
        }

    total_area_ratio = 0.0
    max_weighted_area_ratio = 0.0
    weighted_area_ratio = 0.0
    weighted_count = 0.0
    conf_sum = 0.0

    for det in detections:
        area_ratio = float(det.get("area_ratio", 0.0))
        weight = float(det.get("class_weight", 1.0))
        total_area_ratio += area_ratio
        weighted_area_ratio += area_ratio * weight
        max_weighted_area_ratio = max(max_weighted_area_ratio, area_ratio * weight)
        weighted_count += weight
        conf_sum += float(det.get("confidence", 0.0))

    count = len(detections)
    mean_conf = conf_sum / max(1, count)
    density_per_mpx = count / max(1.0, image_area / 1_000_000.0)

    score_01 = (
        0.38 * min(1.0, weighted_area_ratio / 0.040)
        + 0.22 * min(1.0, max_weighted_area_ratio / 0.020)
        + 0.20 * min(1.0, weighted_count / 10.0)
        + 0.12 * min(1.0, density_per_mpx / 6.0)
        + 0.08 * min(1.0, (1.0 - mean_conf) / 0.60)
    )
    severity_score = round(min(100.0, max(0.0, score_01 * 100.0)), 2)

    if severity_score < 12:
        level = "A"
        recommendation = "Mat duong on dinh, theo doi theo chu ky thong thuong."
    elif severity_score < 30:
        level = "B"
        recommendation = "Co dau hieu hu hong nhe, can tang tan suat kiem tra."
    elif severity_score < 55:
        level = "C"
        recommendation = "Can lap ke hoach sua chua cuc bo trong ngan han."
    else:
        level = "D"
        recommendation = "Muc hu hong cao, uu tien xu ly khan cap."

    return {
        "severity_score": severity_score,
        "severity_level": level,
        "recommendation": recommendation,
        "metrics": {
            "count": count,
            "mean_conf": round(mean_conf, 4),
            "total_area_ratio": round(total_area_ratio, 6),
            "max_area_ratio": round(max_weighted_area_ratio, 6),
            "weighted_count": round(weighted_count, 4),
            "weighted_area_ratio": round(weighted_area_ratio, 6),
            "density_per_mpx": round(density_per_mpx, 4),
        },
    }


def build_quality_assessment(
    raw_count: int,
    final_count: int,
    deep_count: int,
    basic_count: int,
    fused_added: int,
    filter_stats: dict[str, int],
    max_conf: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    keep_ratio = (final_count / raw_count) if raw_count > 0 else 1.0

    if raw_count > 0 and final_count == 0:
        reasons.append("all_boxes_filtered")
    if raw_count > 0 and keep_ratio < clamp_float(QA_MIN_KEEP_RATIO, 0.0, 1.0):
        reasons.append("heavy_filtering")
    if deep_count == 0 and basic_count > 0:
        reasons.append("deep_miss_basic_hit")
    if fused_added > 0:
        reasons.append("basic_fallback_used")
    if final_count > 0 and max_conf < clamp_float(QA_LOW_CONF_THRESH, 0.05, 0.99):
        reasons.append("low_confidence")
    if filter_stats.get("removed_watermark_zone", 0) > 0:
        reasons.append("logo_or_watermark_suspected")

    needs_review = len(reasons) > 0
    return {
        "needs_review": needs_review,
        "reasons": reasons,
        "keep_ratio": round(keep_ratio, 4),
        "max_conf": round(max_conf, 4),
    }


def run_prediction_with_timeout(
    model_path: Path,
    image_bgr: np.ndarray,
    fallback_names: dict[int, str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    timeout_ms: int,
) -> dict[str, Any] | None:
    global _assist_cooldown_until
    now = _now_s()
    with _assist_state_lock:
        if now < _assist_cooldown_until:
            return None

    future = _assist_executor.submit(
        run_prediction,
        model_path,
        image_bgr,
        fallback_names,
        conf,
        iou,
        imgsz,
        device,
    )
    try:
        return future.result(timeout=max(0.05, float(timeout_ms) / 1000.0))
    except FutureTimeoutError:
        future.cancel()
        with _assist_state_lock:
            _assist_cooldown_until = _now_s() + max(0.2, float(timeout_ms) / 1000.0)
        return None
    except Exception:
        return None


def ensure_stream_db() -> None:
    STREAM_STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_ts REAL NOT NULL,
                    last_update_ts REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    session_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    project_id INTEGER NOT NULL DEFAULT 1,
                    stage TEXT NOT NULL,
                    has_crack INTEGER NOT NULL,
                    max_confidence REAL NOT NULL,
                    detections_count INTEGER NOT NULL,
                    infer_time_ms REAL NOT NULL,
                    input_source TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    input_path TEXT NOT NULL DEFAULT '',
                    output_path TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    is_archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            _ensure_analysis_history_columns(conn)
            _ensure_default_project(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_ts
                ON audit_events(ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_type
                ON audit_events(event_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_session
                ON audit_events(session_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analysis_history_ts
                ON analysis_history(ts)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analysis_history_project
                ON analysis_history(project_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_projects_updated
                ON projects(updated_ts)
                """
            )
            conn.commit()
        finally:
            conn.close()


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows if len(r) > 1}


def _ensure_analysis_history_columns(conn: sqlite3.Connection) -> None:
    cols = _get_table_columns(conn, "analysis_history")
    if "project_id" not in cols:
        conn.execute("ALTER TABLE analysis_history ADD COLUMN project_id INTEGER NOT NULL DEFAULT 1")
    if "input_path" not in cols:
        conn.execute("ALTER TABLE analysis_history ADD COLUMN input_path TEXT NOT NULL DEFAULT ''")
    if "output_path" not in cols:
        conn.execute("ALTER TABLE analysis_history ADD COLUMN output_path TEXT NOT NULL DEFAULT ''")


def _ensure_default_project(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT id FROM projects WHERE is_archived = 0 ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if row:
        return int(row[0])
    now = _now_s()
    cur = conn.execute(
        """
        INSERT INTO projects(name, description, created_ts, updated_ts, is_archived)
        VALUES (?, ?, ?, ?, 0)
        """,
        ("Du an mac dinh", "Du an tao tu dong de luu ket qua detect", now, now),
    )
    return int(cur.lastrowid or 1)


def save_stream_session_to_db(session: dict[str, Any]) -> None:
    payload = json.dumps(session, ensure_ascii=True)
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            conn.execute(
                """
                INSERT INTO stream_sessions(session_id, created_ts, last_update_ts, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    created_ts=excluded.created_ts,
                    last_update_ts=excluded.last_update_ts,
                    payload_json=excluded.payload_json
                """,
                (
                    str(session.get("session_id")),
                    float(session.get("created_ts", _now_s())),
                    float(session.get("last_update_ts", _now_s())),
                    payload,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def delete_stream_session_from_db(session_id: str) -> None:
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            conn.execute("DELETE FROM stream_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


def cleanup_stream_db_expired() -> None:
    ttl = max(60, int(STREAM_SESSION_TTL_SEC))
    cutoff = _now_s() - ttl
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            conn.execute("DELETE FROM stream_sessions WHERE last_update_ts < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()


def cleanup_audit_db_overflow() -> None:
    max_events = max(1000, int(AUDIT_DB_MAX_EVENTS))
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            row = conn.execute("SELECT COUNT(1) FROM audit_events").fetchone()
            total = int(row[0]) if row else 0
            overflow = total - max_events
            if overflow <= 0:
                return
            conn.execute(
                """
                DELETE FROM audit_events
                WHERE id IN (
                    SELECT id FROM audit_events ORDER BY id ASC LIMIT ?
                )
                """,
                (overflow,),
            )
            conn.commit()
        finally:
            conn.close()


def append_audit_event(event_type: str, payload: dict[str, Any], session_id: str | None = None) -> None:
    if not AUDIT_LOG_ENABLE:
        return
    safe_payload = dict(payload or {})
    try:
        payload_json = json.dumps(safe_payload, ensure_ascii=True)
    except Exception:
        payload_json = json.dumps({"serialization_error": True}, ensure_ascii=True)

    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            conn.execute(
                """
                INSERT INTO audit_events(ts, event_type, session_id, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (_now_s(), str(event_type), str(session_id) if session_id else None, payload_json),
            )
            conn.commit()
        finally:
            conn.close()

    cleanup_audit_db_overflow()


def query_audit_events(
    limit: int = 100,
    event_type: str = "",
    session_id: str = "",
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(2000, int(limit)))
    clauses: list[str] = []
    args: list[Any] = []

    if event_type:
        clauses.append("event_type = ?")
        args.append(str(event_type))
    if session_id:
        clauses.append("session_id = ?")
        args.append(str(session_id))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT id, ts, event_type, session_id, payload_json "
        f"FROM audit_events {where_sql} ORDER BY id DESC LIMIT ?"
    )
    args.append(safe_limit)

    out: list[dict[str, Any]] = []
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            rows = conn.execute(sql, tuple(args)).fetchall()
        finally:
            conn.close()

    for rid, ts, etype, sid, payload_json in rows:
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {"raw_payload": str(payload_json)}
        out.append(
            {
                "id": int(rid),
                "ts": float(ts),
                "event_type": str(etype),
                "session_id": str(sid) if sid is not None else None,
                "payload": payload,
            }
        )
    return out


def _resolve_project_id(conn: sqlite3.Connection, project_id: int | None = None) -> int:
    if project_id is not None and int(project_id) > 0:
        row = conn.execute(
            "SELECT id FROM projects WHERE id = ? AND is_archived = 0",
            (int(project_id),),
        ).fetchone()
        if row:
            return int(row[0])
    return _ensure_default_project(conn)


def list_projects(limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = max(1, min(1000, int(limit)))
    out: list[dict[str, Any]] = []
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            rows = conn.execute(
                """
                SELECT p.id, p.name, p.description, p.created_ts, p.updated_ts,
                       COALESCE(COUNT(h.id), 0) AS history_count
                FROM projects p
                LEFT JOIN analysis_history h ON h.project_id = p.id
                WHERE p.is_archived = 0
                GROUP BY p.id
                ORDER BY p.updated_ts DESC, p.id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        finally:
            conn.close()
    for row in rows:
        pid, name, desc, created_ts, updated_ts, count = row
        out.append(
            {
                "id": int(pid),
                "name": str(name),
                "description": str(desc or ""),
                "created_ts": float(created_ts),
                "updated_ts": float(updated_ts),
                "history_count": int(count),
            }
        )
    return out


def create_project(name: str, description: str = "") -> dict[str, Any]:
    clean_name = str(name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Project name is required")
    if len(clean_name) > 100:
        raise HTTPException(status_code=400, detail="Project name is too long (max 100 chars)")
    clean_desc = str(description or "").strip()
    now = _now_s()
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            row = conn.execute(
                "SELECT id FROM projects WHERE LOWER(name) = LOWER(?) AND is_archived = 0",
                (clean_name,),
            ).fetchone()
            if row:
                pid = int(row[0])
            else:
                cur = conn.execute(
                    """
                    INSERT INTO projects(name, description, created_ts, updated_ts, is_archived)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (clean_name, clean_desc, now, now),
                )
                pid = int(cur.lastrowid)
            conn.commit()
        finally:
            conn.close()
    return {"id": pid, "name": clean_name, "description": clean_desc}


def get_project(project_id: int | None = None) -> dict[str, Any]:
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            pid = _resolve_project_id(conn, project_id=project_id)
            row = conn.execute(
                """
                SELECT id, name, description, created_ts, updated_ts
                FROM projects
                WHERE id = ?
                """,
                (int(pid),),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    rid, name, desc, created_ts, updated_ts = row
    return {
        "id": int(rid),
        "name": str(name),
        "description": str(desc or ""),
        "created_ts": float(created_ts),
        "updated_ts": float(updated_ts),
    }


def _draw_detections_overlay(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    colors = [
        (239, 68, 68),
        (245, 158, 11),
        (234, 179, 8),
        (132, 204, 22),
        (139, 92, 246),
        (6, 182, 212),
    ]
    for det in detections:
        cls_id = int(det.get("class_id", 0))
        color = colors[abs(cls_id) % len(colors)]
        x1, y1, x2, y2 = [int(round(v)) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
        conf = normalize_confidence(det.get("confidence", 0.0)) * 100.0
        name = str(det.get("class_name", f"class_{cls_id}"))
        label = f"{name} ({conf:.1f}%)"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 10, ty + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 5, ty + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def save_project_artifacts(
    project_id: int,
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> tuple[str, str]:
    now_ms = int(_now_s() * 1000)
    token = uuid.uuid4().hex[:8]
    base_dir = PROJECT_STORAGE_ROOT / f"project_{int(project_id)}"
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{now_ms}_{token}"
    input_path = input_dir / f"{stem}_input.jpg"
    output_path = output_dir / f"{stem}_output.jpg"
    cv2.imwrite(str(input_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    rendered = _draw_detections_overlay(image_bgr, detections)
    cv2.imwrite(str(output_path), rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    try:
        input_rel = input_path.relative_to(ROOT).as_posix()
        output_rel = output_path.relative_to(ROOT).as_posix()
    except Exception:
        input_rel = str(input_path)
        output_rel = str(output_path)
    return input_rel, output_rel


def append_analysis_history(entry: dict[str, Any], project_id: int | None = None) -> int:
    payload = dict(entry or {})
    stage = str(payload.get("stage", "deep"))
    has_crack = bool(payload.get("has_crack", False))
    max_conf = normalize_confidence(payload.get("max_confidence", 0.0))
    detections_count = max(0, int(payload.get("detections_count", 0)))
    infer_time_ms = max(0.0, float(payload.get("infer_time_ms", 0.0)))
    input_source = str(payload.get("input_source", "upload"))
    scene = str(payload.get("scene", "default"))
    input_path = str(payload.get("input_path", ""))
    output_path = str(payload.get("output_path", ""))
    payload_json = json.dumps(payload, ensure_ascii=True)

    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            pid = _resolve_project_id(conn, project_id=project_id)
            cur = conn.execute(
                """
                INSERT INTO analysis_history(
                    ts, project_id, stage, has_crack, max_confidence, detections_count,
                    infer_time_ms, input_source, scene, input_path, output_path, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_s(),
                    int(pid),
                    stage,
                    int(has_crack),
                    float(max_conf),
                    int(detections_count),
                    float(infer_time_ms),
                    input_source,
                    scene,
                    input_path,
                    output_path,
                    payload_json,
                ),
            )
            conn.execute(
                "UPDATE projects SET updated_ts = ? WHERE id = ?",
                (_now_s(), int(pid)),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def query_analysis_history(limit: int = 50, project_id: int | None = None) -> list[dict[str, Any]]:
    safe_limit = max(1, min(1000, int(limit)))
    out: list[dict[str, Any]] = []
    clauses: list[str] = []
    args: list[Any] = []
    if project_id is not None and int(project_id) > 0:
        clauses.append("h.project_id = ?")
        args.append(int(project_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            rows = conn.execute(
                f"""
                SELECT h.id, h.ts, h.project_id, p.name, h.stage, h.has_crack, h.max_confidence,
                       h.detections_count, h.infer_time_ms, h.input_source, h.scene,
                       h.input_path, h.output_path, h.payload_json
                FROM analysis_history h
                LEFT JOIN projects p ON p.id = h.project_id
                {where_sql}
                ORDER BY h.id DESC
                LIMIT ?
                """,
                tuple([*args, safe_limit]),
            ).fetchall()
        finally:
            conn.close()

    for row in rows:
        (
            rid,
            ts,
            pid,
            pname,
            stage,
            has_crack,
            max_conf,
            det_count,
            infer_ms,
            input_source,
            scene,
            input_path,
            output_path,
            payload_json,
        ) = row
        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}
        out.append(
            {
                "id": int(rid),
                "ts": float(ts),
                "project_id": int(pid),
                "project_name": str(pname or f"project_{pid}"),
                "stage": str(stage),
                "has_crack": bool(has_crack),
                "max_confidence": float(max_conf),
                "detections_count": int(det_count),
                "infer_time_ms": float(infer_ms),
                "input_source": str(input_source),
                "scene": str(scene),
                "input_path": str(input_path or ""),
                "output_path": str(output_path or ""),
                "payload": payload,
            }
        )
    return out


def get_history_artifact_path(history_id: int, kind: str) -> Path:
    col = "input_path" if kind == "input" else "output_path"
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            row = conn.execute(
                f"SELECT {col} FROM analysis_history WHERE id = ?",
                (int(history_id),),
            ).fetchone()
        finally:
            conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    p = (ROOT / str(row[0])).resolve()
    root_resolved = ROOT.resolve()
    storage_resolved = PROJECT_STORAGE_ROOT.resolve()
    inside_root = root_resolved in p.parents or p == root_resolved
    inside_storage = storage_resolved in p.parents or p == storage_resolved
    if not (inside_root or inside_storage):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return p


def clear_analysis_history(project_id: int | None = None) -> int:
    clauses: list[str] = []
    args: list[Any] = []
    if project_id is not None and int(project_id) > 0:
        clauses.append("project_id = ?")
        args.append(int(project_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    artifact_paths: list[str] = []
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            rows = conn.execute(
                f"SELECT input_path, output_path FROM analysis_history {where_sql}",
                tuple(args),
            ).fetchall()
            for in_path, out_path in rows:
                if in_path:
                    artifact_paths.append(str(in_path))
                if out_path:
                    artifact_paths.append(str(out_path))
            row = conn.execute(f"SELECT COUNT(1) FROM analysis_history {where_sql}", tuple(args)).fetchone()
            deleted = int(row[0]) if row else 0
            conn.execute(f"DELETE FROM analysis_history {where_sql}", tuple(args))
            conn.commit()
        finally:
            conn.close()
    for rel in artifact_paths:
        try:
            p = (ROOT / rel).resolve()
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            continue
    return deleted


def load_stream_sessions_from_db() -> dict[str, dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    with _stream_db_lock:
        conn = sqlite3.connect(str(STREAM_STATE_DB_PATH))
        try:
            rows = conn.execute("SELECT session_id, payload_json FROM stream_sessions").fetchall()
        finally:
            conn.close()
    for sid, payload_json in rows:
        try:
            data = json.loads(payload_json)
            if isinstance(data, dict) and data.get("session_id"):
                sessions[str(sid)] = data
        except Exception:
            continue
    return sessions


def maybe_persist_session(session: dict[str, Any]) -> None:
    processed = int(session.get("processed_frames", 0))
    skipped = int(session.get("skipped_frames", 0))
    if (processed + skipped) % max(1, int(STREAM_DB_SYNC_EVERY)) == 0:
        save_stream_session_to_db(session)


def _now_s() -> float:
    return time.time()


def _cleanup_stream_sessions_locked() -> None:
    global _stream_update_counter
    now = _now_s()
    ttl = max(60, int(STREAM_SESSION_TTL_SEC))
    stale = [
        sid
        for sid, sess in _stream_sessions.items()
        if now - float(sess.get("last_update_ts", now)) > ttl
    ]
    for sid in stale:
        _stream_sessions.pop(sid, None)
        delete_stream_session_from_db(sid)

    if len(_stream_sessions) <= STREAM_MAX_SESSIONS:
        # Periodic DB cleanup as a fallback.
        _stream_update_counter += 1
        if _stream_update_counter % max(10, int(STREAM_DB_SYNC_EVERY) * 20) == 0:
            cleanup_stream_db_expired()
        return

    ordered = sorted(
        _stream_sessions.items(),
        key=lambda kv: float(kv[1].get("last_update_ts", 0.0)),
    )
    overflow = max(0, len(_stream_sessions) - STREAM_MAX_SESSIONS)
    for sid, _ in ordered[:overflow]:
        _stream_sessions.pop(sid, None)
        delete_stream_session_from_db(sid)


def _new_stream_session() -> dict[str, Any]:
    sid = uuid.uuid4().hex
    now = _now_s()
    sess = {
        "session_id": sid,
        "created_ts": now,
        "last_update_ts": now,
        "processed_frames": 0,
        "skipped_frames": 0,
        "candidate_hits": [],
    }
    _ensure_stream_tracking_state(sess)
    return sess


def _get_stream_session_or_404(session_id: str) -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        return sess


def _append_candidate_hit(
    session: dict[str, Any],
    frame_index: int,
    timestamp_sec: float,
    max_conf: float,
    raw_detections: int,
) -> None:
    hits = session.setdefault("candidate_hits", [])
    hits.append(
        {
            "frame_index": int(frame_index),
            "timestamp_sec": float(timestamp_sec),
            "confidence": float(max_conf),
            "raw_detections": int(raw_detections),
        }
    )
    # Keep memory bounded for very long streams.
    if len(hits) > 5000:
        del hits[: len(hits) - 5000]


def summarize_stream_segments(
    candidate_hits: list[dict[str, Any]],
    gap_sec: float,
    frame_gap: int,
) -> list[dict[str, Any]]:
    if not candidate_hits:
        return []
    ordered = sorted(candidate_hits, key=lambda x: (float(x["timestamp_sec"]), int(x["frame_index"])))
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    max_gap = max(0.1, float(gap_sec))
    max_frame_gap = max(1, int(frame_gap))

    for hit in ordered:
        ts = float(hit["timestamp_sec"])
        fi = int(hit["frame_index"])
        conf = float(hit["confidence"])
        if current is None:
            current = {
                "start_ts": ts,
                "end_ts": ts,
                "start_frame": fi,
                "end_frame": fi,
                "peak_conf": conf,
                "mean_conf_sum": conf,
                "samples": 1,
            }
            continue
        ts_gap = ts - float(current["end_ts"])
        fi_gap = fi - int(current["end_frame"])
        if ts_gap <= max_gap and fi_gap <= max_frame_gap:
            current["end_ts"] = ts
            current["end_frame"] = fi
            current["peak_conf"] = max(float(current["peak_conf"]), conf)
            current["mean_conf_sum"] = float(current["mean_conf_sum"]) + conf
            current["samples"] = int(current["samples"]) + 1
        else:
            current["mean_conf"] = float(current["mean_conf_sum"]) / max(1, int(current["samples"]))
            current["duration_sec"] = max(0.0, float(current["end_ts"]) - float(current["start_ts"]))
            current.pop("mean_conf_sum", None)
            segments.append(current)
            current = {
                "start_ts": ts,
                "end_ts": ts,
                "start_frame": fi,
                "end_frame": fi,
                "peak_conf": conf,
                "mean_conf_sum": conf,
                "samples": 1,
            }

    if current is not None:
        current["mean_conf"] = float(current["mean_conf_sum"]) / max(1, int(current["samples"]))
        current["duration_sec"] = max(0.0, float(current["end_ts"]) - float(current["start_ts"]))
        current.pop("mean_conf_sum", None)
        segments.append(current)

    segments.sort(key=lambda x: (float(x["peak_conf"]), int(x["samples"])), reverse=True)
    return segments


def build_stream_summary_payload(
    session_id: str,
    session_data: dict[str, Any],
    top_k: int = 8,
) -> dict[str, Any]:
    processed_frames = int(session_data.get("processed_frames", 0))
    skipped_frames = int(session_data.get("skipped_frames", 0))
    candidate_hits = list(session_data.get("candidate_hits", []))
    active_tracks = len(session_data.get("tracks", {}))
    total_tracks_created = int(session_data.get("next_track_id", 1)) - 1
    created_ts = float(session_data.get("created_ts", _now_s()))
    last_update_ts = float(session_data.get("last_update_ts", _now_s()))

    segments = summarize_stream_segments(
        candidate_hits,
        gap_sec=clamp_float(STREAM_SEGMENT_GAP_SEC, 0.2, 10.0),
        frame_gap=max(1, STREAM_FRAME_STRIDE * 2),
    )
    top_hits = sorted(candidate_hits, key=lambda x: float(x["confidence"]), reverse=True)[: max(1, int(top_k))]
    recommend_deep_scan = len(segments) > 0

    return {
        "ok": True,
        "session_id": session_id,
        "mode": "basic_stream_then_optional_deep_scan",
        "stats": {
            "processed_frames": processed_frames,
            "skipped_frames": skipped_frames,
            "candidate_frames": len(candidate_hits),
            "segments": len(segments),
            "active_tracks": active_tracks,
            "total_tracks_created": total_tracks_created,
            "created_ts": created_ts,
            "last_update_ts": last_update_ts,
            "uptime_sec": max(0.0, last_update_ts - created_ts),
        },
        "recommend_deep_scan": recommend_deep_scan,
        "segments": segments,
        "top_candidate_frames": top_hits,
    }


def build_stream_product_report(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("stats", {}) if isinstance(summary, dict) else {}
    segments = summary.get("segments", []) if isinstance(summary, dict) else []
    top_frames = summary.get("top_candidate_frames", []) if isinstance(summary, dict) else []

    processed = int(stats.get("processed_frames", 0))
    candidate_frames = int(stats.get("candidate_frames", 0))
    candidate_ratio = (candidate_frames / processed) if processed > 0 else 0.0
    peak_conf = max((float(s.get("peak_conf", 0.0)) for s in segments), default=0.0)
    mean_peak_conf = (
        sum(float(s.get("peak_conf", 0.0)) for s in segments) / len(segments) if len(segments) > 0 else 0.0
    )
    total_segment_duration = sum(float(s.get("duration_sec", 0.0)) for s in segments)
    hottest_frame = top_frames[0] if top_frames else None

    if peak_conf >= 0.85:
        risk_level = "high"
        recommendation = "Uu tien kiem tra hien truong va phan tich chuyen sau ngay."
    elif peak_conf >= 0.65:
        risk_level = "medium"
        recommendation = "Nen quet chuyen sau cac doan co do tin cay cao."
    elif candidate_frames > 0:
        risk_level = "low"
        recommendation = "Co dau hieu nhe, tiep tuc giam sat va quet bo sung."
    else:
        risk_level = "normal"
        recommendation = "Khong phat hien doan nghi ngo trong stream."

    return {
        "generated_ts": _now_s(),
        "session_id": summary.get("session_id"),
        "summary": summary,
        "kpi": {
            "candidate_frame_ratio": round(candidate_ratio, 6),
            "peak_segment_confidence": round(peak_conf, 6),
            "mean_segment_peak_confidence": round(mean_peak_conf, 6),
            "total_segment_duration_sec": round(total_segment_duration, 4),
            "segment_count": len(segments),
            "processed_frames": processed,
        },
        "risk_assessment": {
            "risk_level": risk_level,
            "recommendation": recommendation,
        },
        "hottest_frame": hottest_frame,
        "runtime_policy": {
            "stream_frame_stride": STREAM_FRAME_STRIDE,
            "stream_basic_conf": STREAM_BASIC_CONF,
            "stream_candidate_conf": STREAM_CANDIDATE_CONF,
            "stream_imgsz": STREAM_IMGSZ,
            "stream_iou": STREAM_IOU,
            "stream_track_enable": STREAM_TRACK_ENABLE,
            "stream_track_iou": STREAM_TRACK_IOU,
            "stream_track_max_age": STREAM_TRACK_MAX_AGE,
        },
    }


def _build_csv_bytes(fieldnames: list[str], rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        clean_row = {k: row.get(k) for k in fieldnames}
        writer.writerow(clean_row)
    return buf.getvalue().encode("utf-8")


def build_stream_bundle_bytes(report: dict[str, Any]) -> bytes:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    segments = list(summary.get("segments", [])) if isinstance(summary, dict) else []
    top_frames = list(summary.get("top_candidate_frames", [])) if isinstance(summary, dict) else []

    segment_rows = []
    for idx, seg in enumerate(segments, start=1):
        segment_rows.append(
            {
                "segment_rank": idx,
                "start_ts": float(seg.get("start_ts", 0.0)),
                "end_ts": float(seg.get("end_ts", 0.0)),
                "duration_sec": float(seg.get("duration_sec", 0.0)),
                "start_frame": int(seg.get("start_frame", 0)),
                "end_frame": int(seg.get("end_frame", 0)),
                "peak_conf": float(seg.get("peak_conf", 0.0)),
                "mean_conf": float(seg.get("mean_conf", 0.0)),
                "samples": int(seg.get("samples", 0)),
            }
        )

    top_rows = []
    for idx, hit in enumerate(top_frames, start=1):
        top_rows.append(
            {
                "rank": idx,
                "frame_index": int(hit.get("frame_index", 0)),
                "timestamp_sec": float(hit.get("timestamp_sec", 0.0)),
                "confidence": float(hit.get("confidence", 0.0)),
                "raw_detections": int(hit.get("raw_detections", 0)),
            }
        )

    bundle_buf = io.BytesIO()
    with zipfile.ZipFile(bundle_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))
        zf.writestr(
            "segments.csv",
            _build_csv_bytes(
                [
                    "segment_rank",
                    "start_ts",
                    "end_ts",
                    "duration_sec",
                    "start_frame",
                    "end_frame",
                    "peak_conf",
                    "mean_conf",
                    "samples",
                ],
                segment_rows,
            ),
        )
        zf.writestr(
            "top_candidate_frames.csv",
            _build_csv_bytes(
                [
                    "rank",
                    "frame_index",
                    "timestamp_sec",
                    "confidence",
                    "raw_detections",
                ],
                top_rows,
            ),
        )
    bundle_buf.seek(0)
    return bundle_buf.read()


def _ensure_stream_tracking_state(session: dict[str, Any]) -> None:
    if "tracks" not in session or not isinstance(session.get("tracks"), dict):
        session["tracks"] = {}
    if "next_track_id" not in session:
        session["next_track_id"] = 1
    if "last_tracked_detections" not in session:
        session["last_tracked_detections"] = []
    if "last_processed_frame" not in session:
        session["last_processed_frame"] = -1


def assign_stream_tracks(
    session: dict[str, Any],
    detections: list[dict[str, Any]],
    frame_index: int,
) -> list[dict[str, Any]]:
    if not STREAM_TRACK_ENABLE:
        out = [dict(d) for d in detections]
        session["last_tracked_detections"] = out
        session["last_processed_frame"] = int(frame_index)
        return out

    _ensure_stream_tracking_state(session)
    tracks: dict[str, dict[str, Any]] = session["tracks"]
    next_track_id = int(session["next_track_id"])
    iou_thr = clamp_float(STREAM_TRACK_IOU, 0.05, 0.95)
    max_age = max(1, int(STREAM_TRACK_MAX_AGE))

    active_track_ids = list(tracks.keys())
    used_tracks: set[str] = set()
    out: list[dict[str, Any]] = []

    for det in sorted(detections, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        best_tid = None
        best_iou = 0.0
        for tid in active_track_ids:
            if tid in used_tracks:
                continue
            t = tracks.get(tid)
            if not t:
                continue
            if int(det.get("class_id", -1)) != int(t.get("class_id", -2)):
                continue
            iou_val = bbox_iou_xyxy(det["bbox_xyxy"], t["bbox_xyxy"])
            if iou_val >= iou_thr and iou_val > best_iou:
                best_iou = iou_val
                best_tid = tid

        if best_tid is None:
            tid = str(next_track_id)
            next_track_id += 1
            tracks[tid] = {
                "track_id": int(tid),
                "bbox_xyxy": [float(v) for v in det["bbox_xyxy"]],
                "class_id": int(det.get("class_id", -1)),
                "class_name": str(det.get("class_name", "")),
                "confidence": float(det.get("confidence", 0.0)),
                "last_seen_frame": int(frame_index),
                "age_frames": 0,
            }
            used_tracks.add(tid)
            out.append({**det, "track_id": int(tid), "track_iou": 1.0})
        else:
            t = tracks[best_tid]
            t["bbox_xyxy"] = [float(v) for v in det["bbox_xyxy"]]
            t["class_id"] = int(det.get("class_id", -1))
            t["class_name"] = str(det.get("class_name", ""))
            t["confidence"] = float(det.get("confidence", 0.0))
            t["last_seen_frame"] = int(frame_index)
            t["age_frames"] = 0
            used_tracks.add(best_tid)
            out.append({**det, "track_id": int(best_tid), "track_iou": float(best_iou)})

    # Age or drop unmatched tracks.
    dead_keys: list[str] = []
    for tid, t in tracks.items():
        if tid in used_tracks:
            continue
        t["age_frames"] = int(t.get("age_frames", 0)) + 1
        if int(frame_index) - int(t.get("last_seen_frame", -10_000)) > max_age:
            dead_keys.append(tid)
    for tid in dead_keys:
        tracks.pop(tid, None)

    session["tracks"] = tracks
    session["next_track_id"] = next_track_id
    session["last_tracked_detections"] = out
    session["last_processed_frame"] = int(frame_index)
    return out


def fuse_deep_with_basic(
    deep_dets: list[dict[str, Any]],
    basic_dets: list[dict[str, Any]],
    iou_thr: float,
) -> tuple[list[dict[str, Any]], int]:
    merged = []
    for det in deep_dets:
        merged.append({**det, "source": det.get("source", "deep")})

    added = 0
    for bdet in basic_dets:
        b_box = bdet["bbox_xyxy"]
        overlap = False
        for ddet in deep_dets:
            if bbox_iou_xyxy(b_box, ddet["bbox_xyxy"]) >= iou_thr:
                overlap = True
                break
        if overlap:
            continue
        merged.append(
            {
                "class_id": -1,
                "class_name": "vet nut chua phan loai",
                "confidence": float(bdet["confidence"]),
                "bbox_xyxy": [float(v) for v in b_box],
                "source": "basic_fallback",
            }
        )
        added += 1

    merged.sort(key=lambda x: x["confidence"], reverse=True)
    return merged, added


def clamp_float(v: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, float(v)))


def clamp_int(v: int, min_v: int, max_v: int) -> int:
    return max(min_v, min(max_v, int(v)))


@app.on_event("startup")
def startup_check() -> None:
    if not FRONTEND_DIR.exists():
        raise RuntimeError(f"Missing frontend directory: {FRONTEND_DIR}")
    for required in {MODEL_BASIC_PATH, MODEL_DEEP_PATH}:
        if not required.exists():
            raise RuntimeError(f"Missing model file: {required}")
    ensure_stream_db()
    cleanup_stream_db_expired()
    cleanup_audit_db_overflow()
    with _stream_lock:
        _stream_sessions.clear()
        _stream_sessions.update(load_stream_sessions_from_db())
        for sess in _stream_sessions.values():
            _ensure_stream_tracking_state(sess)
    # Preload weights to avoid first-request latency spikes.
    for model_path in {MODEL_BASIC_PATH, MODEL_DEEP_PATH}:
        load_model_cached(model_path)


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    _assist_executor.shutdown(wait=False, cancel_futures=True)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_5class": str(MODEL_DEEP_PATH),
        "basic_model": str(MODEL_BASIC_PATH),
        "deep_model": str(MODEL_DEEP_PATH),
        "dataset5_yaml": str(DATASET5_YAML_PATH),
        "policy": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "auto_device": AUTO_DEVICE,
            "cuda_available": CUDA_AVAILABLE,
            "use_fp16_on_cuda": USE_FP16_ON_CUDA,
            "basic_conf_min": BASIC_CONF_MIN,
            "basic_pos_conf": BASIC_POS_CONF,
            "basic_review_conf": BASIC_REVIEW_CONF,
            "basic_imgsz_default": BASIC_IMGSZ_DEFAULT,
            "basic_scene_default": BASIC_SCENE,
            "deep_conf_default": DEEP_CONF_DEFAULT,
            "deep_imgsz_default": DEEP_IMGSZ_DEFAULT,
            "deep_scene_default": DEEP_SCENE,
            "deep_enable_tta": DEEP_ENABLE_TTA,
            "deep_tta_scales": DEEP_TTA_SCALES,
            "deep_tta_hflip": DEEP_TTA_HFLIP,
            "deep_enable_tiling": DEEP_ENABLE_TILING,
            "deep_tile_size": DEEP_TILE_SIZE,
            "deep_tile_overlap": DEEP_TILE_OVERLAP,
            "deep_ensemble_nms_iou": DEEP_ENSEMBLE_NMS_IOU,
            "deep_max_return": DEEP_MAX_RETURN,
            "deep_enable_roi_filter": DEEP_ENABLE_ROI_FILTER,
            "deep_roi_top_exclude_ratio": DEEP_ROI_TOP_EXCLUDE_RATIO,
            "deep_enable_watermark_mask": DEEP_ENABLE_WATERMARK_MASK,
            "deep_watermark_right_ratio": DEEP_WATERMARK_RIGHT_RATIO,
            "deep_watermark_bottom_ratio": DEEP_WATERMARK_BOTTOM_RATIO,
            "deep_min_area_px": DEEP_MIN_AREA_PX,
            "deep_min_area_ratio": DEEP_MIN_AREA_RATIO,
            "deep_thin_crack_rescue": DEEP_THIN_CRACK_RESCUE,
            "deep_thin_crack_min_ar": DEEP_THIN_CRACK_MIN_AR,
            "deep_thin_crack_min_conf": DEEP_THIN_CRACK_MIN_CONF,
            "qa_low_conf_thresh": QA_LOW_CONF_THRESH,
            "qa_min_keep_ratio": QA_MIN_KEEP_RATIO,
            "stream_imgsz": STREAM_IMGSZ,
            "stream_iou": STREAM_IOU,
            "stream_basic_conf": STREAM_BASIC_CONF,
            "stream_candidate_conf": STREAM_CANDIDATE_CONF,
            "stream_device": STREAM_DEVICE,
            "stream_frame_stride": STREAM_FRAME_STRIDE,
            "stream_segment_gap_sec": STREAM_SEGMENT_GAP_SEC,
            "stream_track_enable": STREAM_TRACK_ENABLE,
            "stream_track_iou": STREAM_TRACK_IOU,
            "stream_track_max_age": STREAM_TRACK_MAX_AGE,
            "stream_state_db_path": str(STREAM_STATE_DB_PATH),
            "project_storage_root": str(PROJECT_STORAGE_ROOT),
            "stream_db_sync_every": STREAM_DB_SYNC_EVERY,
            "audit_log_enable": AUDIT_LOG_ENABLE,
            "audit_db_max_events": AUDIT_DB_MAX_EVENTS,
            "preprocess_enable": PREPROCESS_ENABLE,
            "preprocess_enable_clahe": PREPROCESS_ENABLE_CLAHE,
        },
    }


@app.get("/api/audit/events")
def audit_events(limit: int = 100, event_type: str = "", session_id: str = "") -> dict[str, Any]:
    events = query_audit_events(limit=limit, event_type=event_type, session_id=session_id)
    return {
        "ok": True,
        "count": len(events),
        "limit": max(1, min(2000, int(limit))),
        "events": events,
    }


@app.get("/api/history")
def analysis_history(limit: int = 50, project_id: int = 0) -> dict[str, Any]:
    pid = int(project_id) if int(project_id) > 0 else None
    rows = query_analysis_history(limit=limit, project_id=pid)
    return {
        "ok": True,
        "count": len(rows),
        "limit": max(1, min(1000, int(limit))),
        "project_id": int(project_id) if int(project_id) > 0 else None,
        "items": rows,
    }


@app.delete("/api/history")
def analysis_history_clear(project_id: int = 0) -> dict[str, Any]:
    pid = int(project_id) if int(project_id) > 0 else None
    deleted = clear_analysis_history(project_id=pid)
    append_audit_event("history_clear", {"deleted": int(deleted), "project_id": pid})
    return {"ok": True, "deleted": int(deleted), "project_id": pid}


@app.get("/api/history/{history_id}/artifact/{kind}")
def analysis_history_artifact(history_id: int, kind: str) -> FileResponse:
    normalized = str(kind).strip().lower()
    if normalized not in {"input", "output"}:
        raise HTTPException(status_code=400, detail="Invalid artifact kind")
    p = get_history_artifact_path(history_id, normalized)
    return FileResponse(path=str(p), filename=p.name)


@app.get("/api/projects")
def projects_list(limit: int = 200) -> dict[str, Any]:
    items = list_projects(limit=limit)
    active_project_id = int(items[0]["id"]) if items else None
    return {"ok": True, "count": len(items), "items": items, "active_project_id": active_project_id}


@app.post("/api/projects")
def projects_create(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project = create_project(payload.get("name", ""), payload.get("description", ""))
    append_audit_event("project_create", {"project_id": int(project["id"]), "name": str(project["name"])})
    return {"ok": True, "project": project}


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/analyze/basic")
def analyze_basic(
    file: UploadFile = File(...),
    conf: float = Form(BASIC_CONF_MIN),
    iou: float = Form(0.6),
    imgsz: int = Form(BASIC_IMGSZ_DEFAULT),
    device: str = Form("auto"),
    input_source: str = Form("upload"),
    scene: str = Form(BASIC_SCENE),
    project_id: int = Form(0),
) -> dict[str, Any]:
    raw_image = decode_upload(file)
    scene_resolved = normalize_scene(scene, raw_image)
    image_bgr = preprocess_image_for_inference(raw_image, input_source, scene_resolved)
    conf = clamp_float(conf, 0.01, 0.99)
    iou = clamp_float(iou, 0.1, 0.95)
    imgsz = clamp_int(imgsz, 320, 1536)
    infer_conf, imgsz = apply_scene_profile("basic", conf, imgsz, scene_resolved)
    t0 = time.perf_counter()

    fallback = get_stage2_names()
    result = run_prediction(MODEL_DEEP_PATH, image_bgr, fallback, infer_conf, iou, imgsz, device)
    detections = postprocess_basic_detections(list(result["detections"]), result["width"], result["height"])
    has_crack = len(detections) > 0
    max_conf = max((float(d["confidence"]) for d in detections), default=0.0)

    pos_thr = clamp_float(BASIC_POS_CONF, 0.05, 0.99)
    review_thr = clamp_float(BASIC_REVIEW_CONF, 0.01, pos_thr)
    if not has_crack:
        triage_status = "negative"
    elif max_conf >= pos_thr:
        triage_status = "positive"
    elif max_conf >= review_thr:
        triage_status = "review"
    else:
        triage_status = "review"

    total_ms = (time.perf_counter() - t0) * 1000.0

    append_audit_event(
        "analyze_basic",
        {
            "project_id": int(project_id) if int(project_id) > 0 else None,
            "has_crack": bool(has_crack),
            "triage_status": str(triage_status),
            "max_confidence": round(float(max_conf), 6),
            "raw_detections": int(len(detections)),
            "scene": str(scene_resolved),
            "input_source": str(input_source),
            "infer_time_ms": round(float(total_ms), 3),
            "device_used": result.get("device_used", "auto"),
            "width": int(result["width"]),
            "height": int(result["height"]),
        },
    )
    project = get_project(project_id if int(project_id) > 0 else None)
    input_path, output_path = save_project_artifacts(project["id"], raw_image, detections)
    history_id = append_analysis_history(
        {
            "stage": "basic5",
            "has_crack": bool(has_crack),
            "max_confidence": float(max_conf),
            "detections_count": int(len(detections)),
            "infer_time_ms": round(float(total_ms), 3),
            "input_source": str(input_source),
            "scene": str(scene_resolved),
            "triage_status": str(triage_status),
            "input_path": input_path,
            "output_path": output_path,
        },
        project_id=project["id"],
    )

    return {
        "ok": True,
        "stage": "basic",
        "history_id": int(history_id),
        "project_id": int(project["id"]),
        "project_name": str(project["name"]),
        "has_crack": has_crack,
        "triage_status": triage_status,
        "max_confidence": max_conf,
        "thresholds": {"positive": pos_thr, "review": review_thr},
        "needs_manual_review": triage_status == "review",
        "detections": detections,
        "raw_detections": len(detections),
        "assist_used": False,
        "assist_raw_detections": 0,
        "assist_timeout": False,
        "infer_time_ms": round(float(total_ms), 2),
        "basic_infer_time_ms": result.get("infer_time_ms", 0.0),
        "device_used": result.get("device_used", "auto"),
        "scene": scene_resolved,
        "input_source": str(input_source),
        "decision_mode": "single_model_5class",
        "policy_version": "single_5class_v1",
        "width": result["width"],
        "height": result["height"],
        "input_path": input_path,
        "output_path": output_path,
        "names": fallback,
    }


@app.post("/api/analyze/deep")
def analyze_deep(
    file: UploadFile = File(...),
    conf: float = Form(DEEP_CONF_DEFAULT),
    iou: float = Form(0.6),
    imgsz: int = Form(DEEP_IMGSZ_DEFAULT),
    device: str = Form("auto"),
    input_source: str = Form("upload"),
    scene: str = Form(DEEP_SCENE),
    project_id: int = Form(0),
) -> dict[str, Any]:
    raw_image = decode_upload(file)
    scene_resolved = normalize_scene(scene, raw_image)
    image_bgr = preprocess_image_for_inference(raw_image, input_source, scene_resolved)
    h, w = image_bgr.shape[:2]
    conf = clamp_float(conf, 0.01, 0.99)
    iou = clamp_float(iou, 0.1, 0.95)
    imgsz = clamp_int(imgsz, 320, 1536)
    conf, imgsz = apply_scene_profile("deep", conf, imgsz, scene_resolved)
    t0 = time.perf_counter()

    fallback = get_stage2_names()
    deep_dets, pass_stats, names_map = run_deep_ensemble(image_bgr, conf, iou, imgsz, device, fallback)
    deep_count = len(deep_dets)

    merged_detections = [{**det, "source": det.get("source", "deep")} for det in deep_dets]
    basic_count = 0
    fused_added = 0

    merged_detections = nms_detections(
        merged_detections,
        iou_thr=clamp_float(DEEP_ENSEMBLE_NMS_IOU, 0.1, 0.95),
        max_keep=clamp_int(DEEP_MAX_RETURN, 10, 1000),
        class_aware=True,
    )
    raw_after_merge = len(merged_detections)

    final_detections, filter_stats = apply_post_filters(merged_detections, w, h)
    final_count = len(final_detections)
    max_conf = max((float(d["confidence"]) for d in final_detections), default=0.0)
    severity = compute_severity(final_detections, w, h)
    class_summary = summarize_by_class(final_detections, image_area=float(w * h))
    quality = build_quality_assessment(
        raw_count=raw_after_merge,
        final_count=final_count,
        deep_count=deep_count,
        basic_count=basic_count,
        fused_added=fused_added,
        filter_stats=filter_stats,
        max_conf=max_conf,
    )
    infer_time_ms = (time.perf_counter() - t0) * 1000.0

    append_audit_event(
        "analyze_deep",
        {
            "project_id": int(project_id) if int(project_id) > 0 else None,
            "raw_detections": int(final_count),
            "raw_before_postfilter": int(raw_after_merge),
            "deep_detections": int(deep_count),
            "basic_detections": int(basic_count),
            "fused_added_from_basic": int(fused_added),
            "severity_score": float(severity["severity_score"]),
            "severity_level": str(severity["severity_level"]),
            "qa_flag": bool(quality["needs_review"]),
            "qa_reasons": list(quality.get("reasons", [])),
            "scene": str(scene_resolved),
            "input_source": str(input_source),
            "infer_time_ms": round(float(infer_time_ms), 3),
            "device_used": normalize_device(device) or AUTO_DEVICE,
            "width": int(w),
            "height": int(h),
        },
    )
    project = get_project(project_id if int(project_id) > 0 else None)
    input_path, output_path = save_project_artifacts(project["id"], raw_image, final_detections)
    history_id = append_analysis_history(
        {
            "stage": "deep5",
            "has_crack": bool(final_count > 0),
            "max_confidence": float(max_conf),
            "detections_count": int(final_count),
            "infer_time_ms": round(float(infer_time_ms), 3),
            "input_source": str(input_source),
            "scene": str(scene_resolved),
            "severity_level": str(severity["severity_level"]),
            "severity_score": float(severity["severity_score"]),
            "qa_flag": bool(quality["needs_review"]),
            "qa_reasons": list(quality.get("reasons", [])),
            "kpi_by_class": class_summary,
            "input_path": input_path,
            "output_path": output_path,
        },
        project_id=project["id"],
    )

    return {
        "ok": True,
        "stage": "deep",
        "history_id": int(history_id),
        "project_id": int(project["id"]),
        "project_name": str(project["name"]),
        "raw_detections": final_count,
        "raw_before_postfilter": raw_after_merge,
        "deep_detections": deep_count,
        "basic_detections": basic_count,
        "fused_added_from_basic": fused_added,
        "fusion_enabled": False,
        "fusion_iou": 0.0,
        "fallback_basic_conf": 0.0,
        "tta_tiling_passes": pass_stats,
        "post_filter_stats": filter_stats,
        "severity_score": severity["severity_score"],
        "severity_level": severity["severity_level"],
        "recommendation": severity["recommendation"],
        "severity_metrics": severity["metrics"],
        "kpi_by_class": class_summary,
        "qa_flag": quality["needs_review"],
        "qa": quality,
        "infer_time_ms": round(float(infer_time_ms), 2),
        "device_used": normalize_device(device) or AUTO_DEVICE,
        "scene": scene_resolved,
        "input_source": str(input_source),
        "decision_mode": "single_model_5class",
        "width": int(w),
        "height": int(h),
        "input_path": input_path,
        "output_path": output_path,
        "names": names_map or fallback,
        "detections": final_detections,
    }


@app.post("/api/stream/session/start")
def stream_session_start() -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _new_stream_session()
        _stream_sessions[sess["session_id"]] = sess
        save_stream_session_to_db(sess)
    append_audit_event(
        "stream_start",
        {
            "frame_stride": max(1, STREAM_FRAME_STRIDE),
            "imgsz": clamp_int(STREAM_IMGSZ, 320, 1536),
            "conf": clamp_float(STREAM_BASIC_CONF, 0.01, 0.99),
            "candidate_conf": clamp_float(STREAM_CANDIDATE_CONF, 0.05, 0.99),
            "iou": clamp_float(STREAM_IOU, 0.1, 0.95),
            "device": STREAM_DEVICE,
        },
        session_id=sess["session_id"],
    )
    return {
        "ok": True,
        "session_id": sess["session_id"],
        "mode": "basic_realtime_only",
        "stream_policy": {
            "frame_stride": max(1, STREAM_FRAME_STRIDE),
            "imgsz": clamp_int(STREAM_IMGSZ, 320, 1536),
            "conf": clamp_float(STREAM_BASIC_CONF, 0.01, 0.99),
            "candidate_conf": clamp_float(STREAM_CANDIDATE_CONF, 0.05, 0.99),
            "iou": clamp_float(STREAM_IOU, 0.1, 0.95),
            "device": STREAM_DEVICE,
            "track_enable": STREAM_TRACK_ENABLE,
            "track_iou": clamp_float(STREAM_TRACK_IOU, 0.05, 0.95),
            "track_max_age": max(1, STREAM_TRACK_MAX_AGE),
        },
    }


@app.post("/api/stream/frame/basic")
def stream_basic_frame(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    frame_index: int = Form(0),
    timestamp_sec: float = Form(0.0),
    force_process: bool = Form(False),
    scene: str = Form("auto"),
) -> dict[str, Any]:
    stride = max(1, int(STREAM_FRAME_STRIDE))
    should_process = bool(force_process) or (int(frame_index) % stride == 0)

    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        _ensure_stream_tracking_state(sess)
        sess["last_update_ts"] = _now_s()
        if not should_process:
            sess["skipped_frames"] = int(sess.get("skipped_frames", 0)) + 1
            cached = list(sess.get("last_tracked_detections", []))
            track_count = len(sess.get("tracks", {}))
            maybe_persist_session(sess)
            return {
                "ok": True,
                "processed": False,
                "session_id": session_id,
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
                "reason": "stride_skip",
                "stride": stride,
                "cached_detections": cached,
                "track_count": track_count,
            }

    raw_image = decode_upload(file)
    scene_resolved = normalize_scene(scene, raw_image)
    image_bgr = preprocess_image_for_inference(raw_image, "stream", scene_resolved)
    conf = clamp_float(STREAM_BASIC_CONF, 0.01, 0.99)
    iou = clamp_float(STREAM_IOU, 0.1, 0.95)
    imgsz = clamp_int(STREAM_IMGSZ, 320, 1536)
    conf, imgsz = apply_scene_profile("basic", conf, imgsz, scene_resolved)
    fallback = get_stage2_names()
    result = run_prediction(MODEL_DEEP_PATH, image_bgr, fallback, conf, iou, imgsz, STREAM_DEVICE)

    detections = postprocess_basic_detections(list(result["detections"]), result["width"], result["height"])
    tracked_detections: list[dict[str, Any]]
    with _stream_lock:
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        _ensure_stream_tracking_state(sess)
        tracked_detections = assign_stream_tracks(sess, detections, int(frame_index))
    has_crack = len(tracked_detections) > 0
    max_conf = max((float(d["confidence"]) for d in tracked_detections), default=0.0)
    candidate = has_crack and max_conf >= clamp_float(STREAM_CANDIDATE_CONF, 0.05, 0.99)

    with _stream_lock:
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        sess["last_update_ts"] = _now_s()
        sess["processed_frames"] = int(sess.get("processed_frames", 0)) + 1
        if candidate:
            _append_candidate_hit(
                sess,
                frame_index=int(frame_index),
                timestamp_sec=float(timestamp_sec),
                max_conf=max_conf,
                raw_detections=len(tracked_detections),
            )
        maybe_persist_session(sess)
        candidate_count = len(sess.get("candidate_hits", []))
        track_count = len(sess.get("tracks", {}))

    if candidate:
        append_audit_event(
            "stream_candidate_hit",
            {
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
                "max_confidence": round(float(max_conf), 6),
                "raw_detections": int(len(tracked_detections)),
                "candidate_total": int(candidate_count),
                "track_count": int(track_count),
                "scene": str(scene_resolved),
            },
            session_id=session_id,
        )

    pos_thr = clamp_float(BASIC_POS_CONF, 0.05, 0.99)
    review_thr = clamp_float(BASIC_REVIEW_CONF, 0.01, pos_thr)
    if not has_crack:
        triage_status = "negative"
    elif max_conf >= pos_thr:
        triage_status = "positive"
    elif max_conf >= review_thr:
        triage_status = "review"
    else:
        triage_status = "review"

    return {
        "ok": True,
        "processed": True,
        "session_id": session_id,
        "frame_index": int(frame_index),
        "timestamp_sec": float(timestamp_sec),
        "has_crack": has_crack,
        "candidate_hit": candidate,
        "triage_status": triage_status,
        "max_confidence": round(max_conf, 4),
        "raw_detections": len(tracked_detections),
        "candidate_total": candidate_count,
        "track_count": track_count,
        "infer_time_ms": result.get("infer_time_ms", 0.0),
        "device_used": result.get("device_used", "auto"),
        "scene": scene_resolved,
        "detections": tracked_detections,
    }


@app.get("/api/stream/session/{session_id}/summary")
def stream_session_summary(session_id: str, top_k: int = 8) -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        sess_copy = json.loads(json.dumps(sess, ensure_ascii=True))

    summary = build_stream_summary_payload(session_id, sess_copy, top_k=max(1, int(top_k)))
    append_audit_event(
        "stream_summary",
        {
            "processed_frames": int(summary["stats"]["processed_frames"]),
            "candidate_frames": int(summary["stats"]["candidate_frames"]),
            "segments": int(summary["stats"]["segments"]),
            "recommend_deep_scan": bool(summary["recommend_deep_scan"]),
            "top_k": max(1, int(top_k)),
        },
        session_id=session_id,
    )
    return summary


@app.get("/api/stream/session/{session_id}/report")
def stream_session_report(session_id: str, top_k: int = 8) -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        sess_copy = json.loads(json.dumps(sess, ensure_ascii=True))
    summary = build_stream_summary_payload(session_id, sess_copy, top_k=max(1, int(top_k)))
    report = build_stream_product_report(summary)
    append_audit_event(
        "stream_report",
        {
            "segment_count": int(report["kpi"]["segment_count"]),
            "candidate_frame_ratio": float(report["kpi"]["candidate_frame_ratio"]),
            "risk_level": str(report["risk_assessment"]["risk_level"]),
        },
        session_id=session_id,
    )
    return {"ok": True, "report": report}


@app.get("/api/stream/session/{session_id}/report/bundle")
def stream_session_report_bundle(session_id: str, top_k: int = 8) -> Response:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        sess_copy = json.loads(json.dumps(sess, ensure_ascii=True))
    summary = build_stream_summary_payload(session_id, sess_copy, top_k=max(1, int(top_k)))
    report = build_stream_product_report(summary)
    payload = build_stream_bundle_bytes(report)
    ts = int(_now_s())
    filename = f"stream_report_{session_id[:8]}_{ts}.zip"
    append_audit_event(
        "stream_report_bundle",
        {
            "segment_count": int(report["kpi"]["segment_count"]),
            "bytes": int(len(payload)),
        },
        session_id=session_id,
    )
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/stream/session/{session_id}/reset")
def stream_session_reset(session_id: str) -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        sess = _stream_sessions.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
        sess["processed_frames"] = 0
        sess["skipped_frames"] = 0
        sess["candidate_hits"] = []
        sess["last_update_ts"] = _now_s()
        sess["tracks"] = {}
        sess["next_track_id"] = 1
        sess["last_tracked_detections"] = []
        sess["last_processed_frame"] = -1
        save_stream_session_to_db(sess)
    append_audit_event("stream_reset", {"reset": True}, session_id=session_id)
    return {"ok": True, "session_id": session_id, "reset": True}


@app.delete("/api/stream/session/{session_id}")
def stream_session_delete(session_id: str) -> dict[str, Any]:
    with _stream_lock:
        _cleanup_stream_sessions_locked()
        removed = _stream_sessions.pop(session_id, None) is not None
        if removed:
            delete_stream_session_from_db(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Unknown stream session: {session_id}")
    append_audit_event("stream_delete", {"deleted": True}, session_id=session_id)
    return {"ok": True, "session_id": session_id, "deleted": True}
