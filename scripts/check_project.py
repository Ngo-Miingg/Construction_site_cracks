from __future__ import annotations

from pathlib import Path
import py_compile
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "models/best5class.pt",
    "configs/dataset5class.yaml",
    "frontend/index.html",
    "frontend/script.js",
    "frontend/style.css",
    "backend/app.py",
]
PY_FILES = [
    "backend/app.py",
    "demo/app_streamlit.py",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("MISSING:")
        for item in missing:
            print(f" - {item}")
        return 1

    for rel in PY_FILES:
        py_compile.compile(str(ROOT / rel), doraise=True)

    print("PROJECT OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except py_compile.PyCompileError as exc:
        print(f"PY_COMPILE_ERROR: {exc.msg}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"CHECK_FAILED: {exc}")
        raise SystemExit(1) from exc
