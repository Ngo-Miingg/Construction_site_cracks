import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from ultralytics import RTDETR


def first_existing(*paths: str) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


DEFAULT_MODEL = first_existing("models/best5class.pt", "best5class.pt")
DEFAULT_DATASET_YAML = first_existing("configs/dataset5class.yaml", "dataset5class.yaml")
DEFAULT_CLASS_NAMES = {
    0: "vet nut doc",
    1: "vet nut ngang",
    2: "nut da ca sau",
    3: "hu hong khac",
    4: "o ga",
}

CLASS_NAME_MAP_VI = {
    "longitudinal crack": "vet nut doc",
    "transverse crack": "vet nut ngang",
    "alligator crack": "nut da ca sau",
    "other corruption": "hu hong khac",
    "pothole": "o ga",
    "crack unclassified": "vet nut chua phan loai",
    "crack_unclassified": "vet nut chua phan loai",
}


@st.cache_resource
def load_model(weights: str, version: int) -> RTDETR:
    _ = version
    w = Path(weights)
    if not w.exists():
        raise FileNotFoundError(f"Missing model file: {w.resolve()}")
    return RTDETR(str(w))


def read_names_from_yaml(dataset_yaml_path: str) -> dict[int, str]:
    ypath = Path(dataset_yaml_path)
    if not ypath.exists():
        return {}
    try:
        data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
    except Exception:
        return {}
    names = data.get("names")
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


def localize_class_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", " ")
    return CLASS_NAME_MAP_VI.get(key, str(name))


def localize_name_map(name_map: dict[int, str]) -> dict[int, str]:
    return {int(k): localize_class_name(v) for k, v in name_map.items()}


def get_model_names(model: RTDETR, fallback: dict[int, str]) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict) and names:
        return localize_name_map({int(k): str(v) for k, v in names.items()})
    return localize_name_map(fallback)


def path_version(path_str: str) -> int:
    p = Path(path_str)
    if not p.exists():
        return 0
    return int(p.stat().st_mtime_ns)


def decode_uploaded_image(uploaded) -> tuple[np.ndarray | None, str]:
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image_bgr, image_hash


def run_inference(
    model: RTDETR,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> tuple[np.ndarray, list[dict]]:
    results = model.predict(
        image_bgr,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device if device else None,
        verbose=False,
    )
    r0 = results[0]
    plotted = r0.plot()
    dets: list[dict] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            dets.append(
                {
                    "cls": int(b.cls.item()) if b.cls is not None else -1,
                    "conf": float(b.conf.item()) if b.conf is not None else 0.0,
                    "x1": float(b.xyxy[0][0].item()),
                    "y1": float(b.xyxy[0][1].item()),
                    "x2": float(b.xyxy[0][2].item()),
                    "y2": float(b.xyxy[0][3].item()),
                }
            )
    return plotted, dets


def make_detection_table(dets: list[dict], names_map: dict[int, str]) -> pd.DataFrame:
    rows = []
    for d in dets:
        rows.append(
            {
                "class_id": d["cls"],
                "class_name": names_map.get(d["cls"], f"class_{d['cls']}"),
                "conf": round(d["conf"], 4),
                "x1": round(d["x1"], 1),
                "y1": round(d["y1"], 1),
                "x2": round(d["x2"], 1),
                "y2": round(d["y2"], 1),
            }
        )
    return pd.DataFrame(rows)


def show_summary(dets: list[dict], names_map: dict[int, str]) -> None:
    if not dets:
        st.info("Khong co box nao duoc phat hien.")
        return
    counter = Counter(d["cls"] for d in dets)
    rows = []
    for cls_id, n in sorted(counter.items(), key=lambda x: x[0]):
        rows.append({"class_id": cls_id, "class_name": names_map.get(cls_id, str(cls_id)), "count": n})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


st.set_page_config(page_title="Road Crack Inspector - 5 class", layout="wide")
st.title("Road Crack Inspector (RT-DETR 5 class)")
st.caption("Ban streamlit nay da toi gian: chi su dung model 5 class.")

st.sidebar.header("Model")
model_path = st.sidebar.text_input("Best 5-class (.pt)", value=DEFAULT_MODEL)
dataset_yaml = st.sidebar.text_input("dataset5class.yaml (optional)", value=DEFAULT_DATASET_YAML)
st.sidebar.caption(f"5-class model: {'OK' if Path(model_path).exists() else 'MISSING'}")

st.sidebar.header("Inference")
imgsz = st.sidebar.select_slider("imgsz", options=[512, 640, 768, 896, 960, 1024], value=896)
conf = st.sidebar.slider("conf", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("iou", 0.1, 0.9, 0.6, 0.05)
device = st.sidebar.text_input("device", value="cpu")

uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
if uploaded is None:
    st.info("Upload 1 image de bat dau phan tich.")
    st.stop()

image_bgr, _image_hash = decode_uploaded_image(uploaded)
if image_bgr is None:
    st.error("Could not decode uploaded image.")
    st.stop()

st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)

if st.button("Phan tich 5 lop", type="primary"):
    try:
        model = load_model(model_path, path_version(model_path))
        names = get_model_names(model, read_names_from_yaml(dataset_yaml) or DEFAULT_CLASS_NAMES)
        out, dets = run_inference(model, image_bgr, conf, iou, imgsz, device.strip())
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.subheader("Ket qua detect")
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="RT-DETR 5-class output", use_container_width=True)
    st.dataframe(make_detection_table(dets, names), use_container_width=True)
    st.markdown("Phan bo class")
    show_summary(dets, names)
