##### ToDo: - class names and masks should appear on the bbox 

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from utils import load_model, preprocess_image

st.title("🔍 Solar Panel Damage Detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("C:\\Users\\panos\\CVision\\Logs\\best_model.pth", device)

uploaded_file = st.file_uploader("Upload an image of a solar panel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    threshold = 0.8
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()

    # Draw boxes
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"Damage: {score:.2f}", fill="white")

    st.image(image, caption="Detected Damages", use_column_width=True)

    # Optional: Display details
    st.write("Detections:")
    for i in range(len(boxes)):
        st.write(f"→ Damage [{i+1}]: Score = {scores[i]:.2f}, Box = {boxes[i].astype(int)}")