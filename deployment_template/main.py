import io 
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = Path("model/cifar10_cnn.pt")
if not MODEL_PATH.exists() and Path("cifar10_cnn.pt").exists():
    MODEL_PATH = Path("cifar10_cnn.pt")

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

app = FastAPI(title="CIFAR-10 Model API")

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not MODEL_PATH.exists():
        return None
    _cached_model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    _cached_model.eval() # switch from training - evalmode
    return _cached_model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException( # if client upload anyfile instead of image
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file.",
        )

    contents = await file.read()
    if len(contents) == 0: #if client upload 0 byte it reject 400
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try: #open ph directly from ram - convert - RGB - resize or shrink
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((32, 32))
    except Exception:
        raise HTTPException(
            status_code=422,  # if file corrupted  - 422
            detail="Corrupted or invalid image file. Unable to decode image bytes.",
        )

    if not MODEL_PATH.exists(): # check if model file is missing 
        return {"error": "Model file not found on server"}

    model = get_model()
    if model is None: # if some one haven't placed the model file yet , return error message instead of crash server
        return {"error": "Model file not found on server"}

    # Preprocess image to tensor, prepare photo for pytouch
    img_array = np.array(image, dtype=np.float32) / 255.0 #normalize pixel 0-255

    img_tensor = np.transpose(img_array, (2, 0, 1)) #re-order dimen form (Height, Width, Colors) to (Colors, Height, Width)
                                                    #pytouch required color first
    tensor = torch.from_numpy(img_tensor).unsqueeze(0) # Adds a batch dimension, making the shape (1, 3, 32, 32): 
                                                    #1 photo, 3 colors, 32 high, 32 wide.

    with torch.no_grad():                           # turn of learning/gradiennt so use les ram and fast
        outputs = model(tensor)                     # pass photo - nn
        probs = torch.softmax(outputs, dim=1)[0].numpy() # output - % probab

    class_id = int(np.argmax(probs))                # find the highest propab
    confidence = float(probs[class_id])

    return {                                        #convert num - word
        "prediction": CLASSES[class_id],            
        "class_id": class_id,
        "confidence": round(confidence, 4),
    }
    # return as json


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
