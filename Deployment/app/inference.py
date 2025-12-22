from PIL import Image, ImageDraw
import torch
import numpy as np
from utils import preprocess_image

def predict(image: Image.Image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]
        threshold = 0.7
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()

    results = []
    for box, label, score in zip(boxes, labels, scores):
        results.append({'label': int(label), 'score': float(score), 'box': box.tolist()})

    return {'predictions': results}