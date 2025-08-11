## ☀️Solar Panel Damage Detector
This project leverages **PyTorch** and **Mask R-CNN** to automatically detect damages on solar panels using image segmentation techniques. A **Streamlit** app is also included to provide an intuitive interface for demo and deployment.
## 📁Project Structure
```
SolarPanel-Damage-Detector/
│
├── src/               # Source code including model training, utils, and inference logic
├── Deployment/        # Streamlit app and deployment-related scripts
├── README.md          # Project overview and usage instructions
├── LICENSE            # MIT License
└── .gitignore         # Files and folders to ignore in Git

```

## 🚀Features
- Damage detection using Mask R-CNN on solar panel images
- Custom dataset loading and preprocessing
- Evaluation metrics and visualization tools
- Streamlit interface for real-time inference
- Open-source

## ⚙️Instalation
```
git clone https://github.com/Ptzatzag/SolarPanel-Damage-Detector.git
cd SolarPanel-Damage-Detector
pip install -r requirements.txt

```

## 🧠Model Training 
Model training is handled using PyTorch and Mask R-CNN. Detailed steps and results are documented in the notebooks/ directory.

## 🌐Streamlit App 
To run the Streamlit app:
```
cd Deployment
streamlit run app.py
```

## 📝License 
This project is licensed under the MIT License. See the LICENSE file for details.  
