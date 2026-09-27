# Week 4 FastAPI Deployment Template: CIFAR-10 CNN Image Classifier

This template serves a trained PyTorch Convolutional Neural Network (CNN) image classification model (from `notebooks/CIFAR_10_CNN.ipynb`) through a FastAPI application.

---

## Setup

```bash
# 1. Navigate into the deployment folder
cd deployment_template

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# macOS, Linux, or Git Bash:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Model Artifact Placement

Save or copy your trained model checkpoint from `notebooks/CIFAR_10_CNN.ipynb` into the `model/` folder as:

```text
model/cifar10_cnn.pt
```

*(Note: The service also supports standard `.keras` and `.h5` model files if you trained with TensorFlow/Keras).*

---

## Running the Service

```bash
uvicorn main:app --reload
```

Open your browser to test interactively:
```text
http://127.0.0.1:8000/docs
```

> **Security Note:** Keep this service on your own computer (`127.0.0.1`). Do not expose it publicly and do not place passwords, API keys, private data, or secrets in this folder.

---

## Expected Checks & Rubric Evidence

### Evidence 1: Health Check (`GET /health`)
- Open `GET /health` in Swagger UI or run `curl http://127.0.0.1:8000/health`.
- When the model is present, it returns HTTP 200:
  ```json
  {
    "status": "ok",
    "model_file_exists": true
  }
  ```
- If the model is not yet placed, it returns `"model_file_exists": false` without crashing.

### Evidence 2: Valid Prediction Request (`POST /predict`)
- Open `POST /predict`, click **Try it out**, and upload a valid image file (e.g., PNG/JPEG).
- Returns HTTP 200 with the predicted class label, class ID, and confidence score:
  ```json
  {
    "prediction": "frog",
    "class_id": 6,
    "confidence": 0.8924
  }
  ```

### Evidence 3: Invalid Request Refusal (Strict Validation)
- **Non-image file rejection**: Uploading a `.txt` or `.csv` file returns **HTTP 400 Bad Request**:
  ```json
  {"detail": "Invalid file type 'text/plain'. Please upload an image file."}
  ```
- **Empty file rejection**: Uploading a 0-byte file returns **HTTP 400 Bad Request**:
  ```json
  {"detail": "Empty file uploaded."}
  ```
- **Corrupted image rejection**: Uploading unreadable or corrupted image bytes returns **HTTP 422 Unprocessable Entity**:
  ```json
  {"detail": "Corrupted or invalid image file. Unable to decode image bytes."}
  ```

### Evidence 4: Behavior Persists After Clean Restart
- Stop the server with `Ctrl+C`.
- Relaunch the server:
  ```bash
  uvicorn main:app --reload
  ```
- Re-run requests to `/health` and `/predict`. Both exhibit identical, deterministic behavior, confirming the service depends on the saved model artifact rather than leftover in-memory session state.

### Graceful Limitation Handling
If `model/cifar10_cnn.pt` is missing, `/predict` returns **HTTP 200** carrying:
```json
{
  "error": "Model file not found on server"
}
```
This proves the service ran and handled the missing model gracefully.

---

## Automated Verification Script

To automatically run and verify all 4 pieces of evidence against the running service, open a second terminal with `.venv` active and run:

```bash
python test_api.py
```
