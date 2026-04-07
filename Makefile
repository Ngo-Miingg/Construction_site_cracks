.PHONY: install run-web run-streamlit check-models check-project

install:
	pip install -U pip
	pip install -r requirements.txt

run-web:
	uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

run-streamlit:
	streamlit run demo/app_streamlit.py

check-models:
	python -c "from pathlib import Path; req=['models/best5class.pt']; miss=[p for p in req if not Path(p).exists()]; print('OK' if not miss else f'MISSING: {miss}')"

check-project:
	python scripts/check_project.py
