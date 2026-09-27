# Week 4 FastAPI Deployment Template

This template serves a trained Scikit-Learn model through a small FastAPI app.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# macOS, Linux, or Git Bash:
source .venv/bin/activate
pip install -r requirements.txt
```

Copy your saved model from the Week 4 notebook into this folder as:

```text
week4_model.joblib
```

Run:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Keep this service on your own computer. Do not expose it publicly and do not
place passwords, API keys, private data, or secrets in this folder.

## Expected checks

1. Open `GET /health`. It should return `"model_file_exists": true`.
2. Open `POST /predict`, select **Try it out**, and send a complete request
   with all five fields, for example:

```json
{
  "mean_radius": 14.0,
  "mean_texture": 20.0,
  "mean_perimeter": 90.0,
  "mean_area": 600.0,
  "mean_smoothness": 0.1
}
```

3. Remove one field and send the request again. The service should reject the
   incomplete request instead of sending it to the model.

If `/health` reports a missing model, return to the notebook, save
`week4_model.joblib`, and copy it into this folder. If `uvicorn` does not
start, stop an earlier server process, confirm the virtual environment is
active, and read the first reported error before changing code.

## Student TODOs

1. Replace the sample `PredictionRequest` fields with fields that match your model.
2. Confirm `MODEL_PATH` points to your saved model.
3. Test `/health`.
4. Test `/predict` from the automatic docs page.
5. Record one valid and one invalid request result, then ask a partner to
   repeat the local checks from a clean restart.

