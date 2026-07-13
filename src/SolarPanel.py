import torch 
import wandb
from configs.configs import SolarConfig, InferenceConfig
from model.maskrcnn import get_model
from dataset.dataset import SolarDataset
from train.train import train
from utils.utils import detect_and_color_splash_pytorch
from PIL import Image
from torchvision import transforms
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# num_classes = 1 + 2  # background + solar damage classes
model = get_model(SolarConfig.num_classes)

# Load the best models weights, for the second cycle
checkpoint = torch.load(SolarConfig.LOGS + "/best_model.pth", map_location=device)
model.load_state_dict(checkpoint)

model.to(device)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")

param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
model_size_bytes = param_size + buffer_size

print(f"Model size: {model_size_bytes / 1024**3:.2f} GB")

# Parse command line arguments
parser = argparse.ArgumentParser(
description='Train Mask R-CNN to detect Solar Panels Damages'
)

parser.add_argument('--dataset',
                    required=False,
                    metavar=SolarConfig.IMAGE_DATA_DIR,
                    help='Root directory of our dataset',
                    default=SolarConfig.IMAGE_DATA_DIR
                    )

parser.add_argument('--weights',
                    required=False,
                    help='Path to weights .pth file or "coco" ',
                    default=SolarConfig.WEIGHTS
                    )

parser.add_argument('--logs',
                    required=False,
                    metavar=SolarConfig.LOGS,
                    help='Path to logs and checkpoints',
                    default=SolarConfig.LOGS
                    )


args = parser.parse_args()   # parser.parse_args(['--dataset', 'pass the path that the dataset is located']), alternative way to preset the value of the argument or we could use default

print("Weights: ", args.weights)
print("Dataset: ", args.dataset)
print("Logs: ", args.logs)


assert args.dataset, "Argument --dataset is required for training"
# Prepare datasets
dataset_train = SolarDataset(dataset_dir=SolarConfig.IMAGE_DATA_DIR, annotation_dir=SolarConfig.ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=True), mode="train", val_size=0.2)
dataset_val = SolarDataset(dataset_dir=SolarConfig.IMAGE_DATA_DIR, annotation_dir=SolarConfig.ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=False), mode="val", val_size=0.2)

wandb.init(project='SolarPanel-Damage-Detector',
            name=f"Snow_FixLoading",
            )
wandb.watch(model, log="gradients", log_freq=30)

train(model, dataset_train, dataset_val, device, activate_l4=60, activate_l3=100, activate_l2=150)