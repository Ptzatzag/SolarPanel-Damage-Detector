## Solar Panel Damage Detector
This project leverages **PyTorch** and **Mask R-CNN** to automatically detect damages on solar panels using image segmentation techniques. This pipeline follows a **transfer learning** strategy, where the model is first trained to detect clean solar panels and is then **fine tuned** using the best checkpoint to recognize snow covered panels. The system currently supports two classes (clean and snow), but it is fully scalable and can be easily extended to additional damage types. Finally a **Streamlit** app is also included to provide an intuitive interface for demo and deployment.

## 🚀Features
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

## Running the application
### 1. Start the FastAPI backend:
```
cd Deployment
uvicorn app.main:app --reload
```
### 2. Start the Streamlit frontend
```
cd Deployment/frontend
streamkit run SLapp.py
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
