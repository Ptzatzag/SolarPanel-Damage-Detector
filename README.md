## Solar Panel Damage Detector
This project uses **PyTorch** and **Mask R-CNN** to automatically detect and localize damage on solar panels from images. The model follows a two-stage transfer-learning strategy. In the first stage, a COCO-pretrained Mask R-CNN is fine-tuned to detect clean solar panels, allowing the network to adapt from general object features to the solar-panel domain. The resulting checkpoint is then used to initialize a second training stage, where the model is further fine-tuned to detect panel conditions and damage such as snow coverage.

The system currently supports two classes (clean and snow), but is designed to be easily extended to additional types of damage. It also includes a **FastAPI** backend for serving predictions and a **Streamlit** frontend for interactive visualization and demo purposes.

## Features
- Damage detection using Mask R-CNN on solar panel images
- Custom dataset loading and preprocessing
- Evaluation metrics and visualization tools
- Streamlit interface for real-time inference
- Open source

## Installation
```
git clone https://github.com/Ptzatzag/SolarPanel-Damage-Detector.git
cd SolarPanel-Damage-Detector
pip install -r requirements.txt

```

## Running the application
### 1. Start the FastAPI backend:
```
uvicorn Deployment.app.main:app --reload
```
### 2. Start the Streamlit frontend
```
cd Deployment/frontend
streamlit run SLapp.py
```
Make sure the backend is running before starting the frontend

## Docker Deployment (Build images and run containers )
```
cd Deployment
docker compose up --build
```
## Example Output
![image](/Examples/CleanExample.PNG)
![image](/Examples/SnowExample.PNG)

## License 
This project is licensed under the MIT License. See the LICENSE file for details.  
