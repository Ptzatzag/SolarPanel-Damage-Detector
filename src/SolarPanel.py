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
num_classes = 1 + 2  # background + 6 solar damage classes
model = get_model(num_classes)

# Load the best models weights, for the second cycle
checkpoint = torch.load("/content/drive/MyDrive/CVision/Logs/best_model.pth", map_location=device)
model.load_state_dict(checkpoint)
# # Remove heads from the checkpoint before loading
# filtered_state_dict = {k: v for k, v in checkpoint.items()
#                        if not ("box_predictor" in k or "mask_predictor" in k)}
# model.load_state_dict(filtered_state_dict, strict=False)

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

parser.add_argument('--mode',   # convert it to a positional argument in the .py file
                    help='train or inference',
                    required=False,
                    default='train'
                    )

parser.add_argument('--dataset',
                    required=False,
                    metavar=SolarConfig.IMAGE_DATA_DIR,
                    help='Root directory of our dataset',
                    default=SolarConfig.IMAGE_DATA_DIR
                    )
### THE FOLLOWING IS SHOULD BE COMMEND OUT ###
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

parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on',
                        default=SolarConfig.IMAGE_INF_EXAMPLE
                    )

args = parser.parse_args()   # parser.parse_args(['--dataset', 'pass the path that the dataset is located']), alternative way to preset the value of the argument or we could use default


# Validate arguments
if args.mode == "train":
    assert args.dataset, "Argument --dataset is required for training"
elif args.mode == "splash":
    assert args.image or args.video,\
            "Provide --image or --video to apply color splash"

print("Weights: ", args.weights)
print("Dataset: ", args.dataset)
print("Logs: ", args.logs)

# Configurations
if args.mode == "train":
    config = SolarConfig()
    assert args.dataset, "Argument --dataset is required for training"
    #config = SolarConfig()
    # Prepare datasets
    dataset_train = SolarDataset(dataset_dir=SolarConfig.IMAGE_DATA_DIR, annotation_dir=SolarConfig.ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=True), mode="train", val_size=0.2)
    dataset_val = SolarDataset(dataset_dir=SolarConfig.IMAGE_DATA_DIR, annotation_dir=SolarConfig.ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=False), mode="val", val_size=0.2)

    wandb.init(project='SolarPanel-Damage-Detector',
                name=f"Snow_FixLoading",
                )
    wandb.watch(model, log="gradients", log_freq=30)

    # train(model, dataset_train, dataset_val, device, activate_backbone=30)
    train(model, dataset_train, dataset_val, device, activate_l4=60, activate_l3=100, activate_l2=150)
else:
    assert args.image, "Provide --image"

    class InferenceConfig(SolarConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    #config.display()

    print("Weights: ", args.weights)   # Think if I should add logger instead of print
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Load model weights
    model.load_state_dict(torch.load(args.weights)) ##########
    model.to(device)
    model.eval()


    # Load image
    image = Image.open(args.image).convert("RGB")

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load and preprocess image
    image_tensor = preprocess(image).to(device)  # e.g. transforms.ToTensor()

    with torch.no_grad():
        outputs = model([image_tensor])

        detect_and_color_splash_pytorch(model, args.image, device)