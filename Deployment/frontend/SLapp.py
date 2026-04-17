# ToDo: draw masks function


import streamlit as st
import requests
from PIL import Image, ImageDraw

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Solar Panel Damage Detector", layout="wide")
st.title("Solar Panel Damage Detector")

def draw_detections(image: Image.Image, detections_raw):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = detections_raw.get('predictions', [])
    
    for det in detections:
        bbox = det.get('box')
        label = det.get('label')
        score = det.get('score')
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            text = label
            if score is not None:
                text = f"{label}: {score:.2f}"

            draw.text((x1, max(y1 - 15, 0)), text, fill="red")

    return annotated

uploaded_file = st.file_uploader(
    "Upload a solar panel image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run detection"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        try:
            with st.spinner("Running inference..."):
                response = requests.post(API_URL, files=files, timeout=60)

            if response.status_code == 200:
                result = response.json()
                
                st.success("Inference completed")
                annotated = draw_detections(image, result)
                st.subheader("Prediction result")
                st.image(annotated, caption="Detected damge", use_column_width=True)
                
                st.subheader("Raw prediction result")
                st.json(result)
            else:
                st.error(f"API error {response.status_code}")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure the API server is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")