import os
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

MODEL_URL = "https://jpxhvgle1grylhsa.public.blob.vercel-storage.com/alzheimers_model-yRRJ7jNvCPbqJNUJqXLchs6OmsPg5Q.h5"
MODEL_PATH = "alzheimers_model.h5"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from blob storage...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Alzheimer's MRI Risk Predictor API is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        result = "Yes" if prediction > 0.5 else "No"
        risk_percent = round(prediction * 100, 2)

        return {
            "alzheimers_detected": result,
            "risk_percentage": risk_percent
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
 
