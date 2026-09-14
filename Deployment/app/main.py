from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from huggingface_hub import hf_hub_download
import torch
from .inference import predict
from model.maskrcnn import load_model
from contextlib import asynccontextmanager
from fastapi import Request



@asynccontextmanager
async def lifespan(app: FastAPI):
    weight_path = hf_hub_download(
        repo_id="Ptzatzag/solar-panel-detector",
        filename="best_modelmult.pth",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = load_model(weight_path, device)

    yield  

    del app.state.model 


app = FastAPI(
    title="Solar Panel Damage Detection",
    lifespan=lifespan,
)

@app.get('/')
def health_check():
    return {'status': 'ok'}

@app.post('/predict')
async def predict_damage(request: Request, file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict(image, request.app.state.model)
    return result