## Solar Panel Damage Detector
This project uses **PyTorch** and **Mask R-CNN** to automatically detect and localize damage on solar panels from images. It follows a transfer learning approach, where the model is initially trained to identify clean panels and then fine tuned to detect snow covered panels.

The system currently supports two classes (clean and snow), but is designed to be easily extended to additional types of damage. It also includes a **FastAPI** backend for serving predictions and a **Streamlit** frontend for interactive visualization and demo purposes.

## 📌Features
- Damage detection using Mask R-CNN on solar panel images
- Custom dataset loading and preprocessing
- Evaluation metrics and visualization tools
- Streamlit interface for real-time inference
- Open source

## ⚙️Instalation
```
git clone https://github.com/Ptzatzag/SolarPanel-Damage-Detector.git
cd SolarPanel-Damage-Detector
pip install -r requirements.txt

```

## 🚀Running the application
### 1. Start the FastAPI backend:
```
cd Deployment
uvicorn app.main:app --reload
```
### 2. Start the Streamlit frontend
```
cd Deployment/frontend
streamlit run SLapp.py
```
Make sure the backend is running before starting the frontend

## 🐳Docker Deployment
```
# Create the docker image
docker build -t solar-panel-damage-detector .
# Run the container
docker run -d -p 8501:8501 solar-panel-damage-detector:latest
```
## Example Output
![image](/Examples/CleanExample.PNG)
![image](/Examples/SnowExample.PNG)

## 📝License 
This project is licensed under the MIT License. See the LICENSE file for details.  
