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


## Model Performance

The final model was fine-tuned for solar-panel condition detection using the Stage 1 solar-panel detector as initialization.

### Evaluation Results

| Metric | Value |
| --- | ---: |
| Bounding Box mAP@50:95 | **0.378** |
| Segmentation Mask mAP@50:95 | **0.415** |
| Training Loss | 0.717 |
| Validation Loss | 1.165 |
| Reported Epoch | 87 |
| Learning Rate | 5.38 × 10⁻⁴ |

The reported mAP values use the COCO evaluation metric averaged across IoU thresholds from 0.50 to 0.95.

### Training Hardware

| Component | Specification |
| --- | --- |
| CPU Cores | 64 |
| Logical CPU Cores | 128 |
| GPU Count | 1 |
| GPU | NVIDIA RTX PRO 4500 Blackwell Server Edition |

### Training Configuration

- Model: Mask R-CNN with ResNet-50 FPN backbone
- Input resolution: 512 × 512
- Mixed-precision training (AMP)
- Gradient accumulation for a larger effective batch size
- AdamW optimizer
- Learning-rate warmup followed by cosine decay
- Progressive backbone unfreezing during fine-tuning
  
## Inference Examples
### Clean Solar Panel Detection
![image](/Examples/CleanExample.PNG)
### Snow-Covered Panel Detection
![image](/Examples/SnowExample.PNG)

The model detects snow-covered solar panels, although segmentation accuracy remains more challenging for this class. Performance can vary depending on snow coverage, lighting conditions, and panel visibility.

## License 
This project is licensed under the MIT License. See the LICENSE file for details.  
