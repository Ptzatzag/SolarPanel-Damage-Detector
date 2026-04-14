import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Solar Panel Damage Detector", layout="wide")
st.title("Solar Panel Damage Detector")

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
                st.subheader("Prediction result")
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