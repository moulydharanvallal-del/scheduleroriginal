# Mini Manufacturing Scheduler (shareable)

This is a lightweight app version of your notebook:
- `scheduler_core.py` contains the scheduling logic + default hardcoded inputs.
- `app.py` is a Streamlit UI so others can run it without touching code.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What to edit
Open the app and edit:
- Customer Orders (JSON)
- BOM / routing data (JSON)
- Work-center capacity (JSON)

## Notes
- Intentionally minimal: no database, no auth, no background jobs.
- If you later want multi-user + API + persistence, we can convert this to FastAPI + a tiny frontend.
