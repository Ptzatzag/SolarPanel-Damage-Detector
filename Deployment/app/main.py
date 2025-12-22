from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from huggingface_hub import hf_hub_download
import torch
from app.inference import predict
from utils import load_model

# Download weights from HF Hub
weight_path = hf_hub_download(
    repo_id="Ptzatzag/solar-panel-detector", 
    filename="best_modelmult.pth"             
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(weight_path, device)


app = FastAPI(title="Snow Detection")

@app.get('/')
def health_check():
    return {'status': 'ok'}

@app.post('/predict')
async def predict_snow(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    
    result = predict(image, model)
    return result