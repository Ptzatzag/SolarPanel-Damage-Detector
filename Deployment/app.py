##### ToDo: - class names and masks should appear on the bbox 

########## INFARENCE SHOULD INCLUDE BOTH CATEGORIES ##########

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from utils import load_model, preprocess_image
from huggingface_hub import hf_hub_download

# Download weights from HF Hub
weight_path = hf_hub_download(
    repo_id="Ptzatzag/solar-panel-detector", 
    # filename="Best_Model_Clean_CC.pth" , 
    filename="best_modelmult.pth"             
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(weight_path, device)

st.title("🔍 Snow On Solar Panel")

uploaded_file = st.file_uploader("Upload an image of a solar panel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    threshold = 0.7
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    masks = output['masks'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    
    # Convert to NumPy
    image_np = np.array(image)

    # ---- DRAW MASKS FIRST ----
    for mask in masks:
        mask = mask[0] > 0.5

        color = np.zeros_like(image_np)
        color[:, :, 2] = 255  # Blue mask

        alpha = 0.3
        image_np[mask] = (
            alpha * color[mask] + (1 - alpha) * image_np[mask]
        ).astype(np.uint8)


    # Draw boxes
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Snow: {score:.2f}", fill="black")

    st.image(image, caption="Detected Snow On Solar Panels", use_column_width=True)

    # Optional: Display details
    st.write("Detections:")
    for i in range(len(boxes)):
        st.write(f"→ Damage [{i+1}]: Score = {scores[i]:.2f}, Box = {boxes[i].astype(int)}")
        # st.write(f"→ Solar Panel [{i+1}]: Score = {scores[i]:.2f}, Box = {boxes[i].astype(int)}")
