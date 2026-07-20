import torch 
import wandb
from configs.configs import SolarConfig
from model.maskrcnn import get_model
from dataset.dataset import SolarDataset
from train.train import train
from PIL import Image
import argparse

config = SolarConfig()  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(config.num_classes)

# Load the best models weights, for the second cycle
# checkpoint = torch.load(config.logs_dir + "/best_model.pth", map_location=device)
checkpoint_path = config.download_weights()  # Download the weights from HF Hub

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)
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
                    metavar=config.image_data_dir,
                    help='Root directory of our dataset',
                    default=config.image_data_dir
                    )

parser.add_argument('--weights',
                    required=False,
                    help='Path to weights .pth file or "coco" ',
                    default=config.weights_path   # needs to be updated 
                    )

parser.add_argument('--logs',
                    required=False,
                    metavar=config.logs_dir,
                    help='Path to logs and checkpoints',
                    default=config.logs_dir
                    )


args = parser.parse_args()   # parser.parse_args(['--dataset', 'pass the path that the dataset is located']), alternative way to preset the value of the argument or we could use default

# print("Weights: ", args.weights)
print("Dataset: ", args.dataset)
print("Logs: ", args.logs)


assert args.dataset, "Argument --dataset is required for training"
# Prepare datasets
dataset_train = SolarDataset(dataset_dir=config.image_data_dir, annotation_dir=config.annotation_json_path, transforms=SolarDataset._get_albumentations_transforms(train=True), mode="train", val_size=0.2)
dataset_val = SolarDataset(dataset_dir=config.image_data_dir, annotation_dir=config.annotation_json_path, transforms=SolarDataset._get_albumentations_transforms(train=False), mode="val", val_size=0.2)

if device.type == 'cpu':
    print('Using CPU, and small dataset for testing purposes')
    train_limit = min(10, len(dataset_train))
    val_limit = min(5, len(dataset_val))

    dataset_train.image_ids = dataset_train.image_ids[:train_limit]
    dataset_train.image_infos = dataset_train.image_infos[:train_limit]
    dataset_train.annotation_info = dataset_train.annotation_info[:train_limit]

    dataset_val.image_ids = dataset_val.image_ids[:val_limit]
    dataset_val.image_infos = dataset_val.image_infos[:val_limit]
    dataset_val.annotation_info = dataset_val.annotation_info[:val_limit]

wandb.init(project='SolarPanel-Damage-Detector',
            name=f"Snow",
            )
wandb.watch(model, log="gradients", log_freq=30)

train(model, dataset_train, dataset_val, device, activate_l4=60, activate_l3=100, activate_l2=150)