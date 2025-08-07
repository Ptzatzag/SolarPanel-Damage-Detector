## ToDo trim the code with config

import torch 
import numpy as np
import json
import sys 
import os 
import torchvision.transforms as T
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import skimage.io
import skimage.draw
from sklearn.model_selection import train_test_split
from typing import Optional, Callable
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import math
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime    
import sys
import argparse
import wandb
import time


class SolarDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 annotation_dir: str,
                 transforms:Optional[Callable]= None,
                 mode: str = "train",
                 val_size: float = 0.2) -> None:
        self.dataset_dir = dataset_dir
        self.annatation_dir = annotation_dir
        self.transforms = transforms
        self.mode = mode
        self.val_size = val_size
    
        with open(annotation_dir) as f:
            annotations_dict = json.load(f)
            
        self.image_infos = list(annotations_dict.values())
        self.image_ids = list(annotations_dict.keys())

        # Optional: Split into train/val here
        if mode in ["train", "val"]:
            from sklearn.model_selection import train_test_split
            train_ids, val_ids = train_test_split(self.image_ids, test_size=val_size, random_state=99)
            self.image_ids = train_ids if mode == "train" else val_ids
            self.image_infos = [annotations_dict[k] for k in self.image_ids]

        # Filter out images with no regions
        self.image_infos = [info for info in self.image_infos if info.get("regions")]
        
    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, idx):
        info = self.image_infos[idx]
        filename = info["filename"]
        image_path = self._resolve_image_path(info)

        #image = skimage.io.imread(image_path)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
            
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        height, width = image.shape[:2]

        masks, class_ids, boxes = self._generate_instance_masks(info, height, width)

        target = {
            "boxes": boxes,
            "labels": class_ids,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            transformed = self.transforms(
                image=image,
                masks=masks.numpy(),
                bboxes=boxes.tolist(),
                labels=class_ids.tolist()
            )
            image = transformed["image"]                         # Already a tensor
            masks = transformed["masks"]                         # Already a list of tensors
            boxes = torch.tensor(transformed["bboxes"])          # Still need to convert
            class_ids = torch.tensor(transformed["labels"], dtype=torch.int64)

            target.update({
                "boxes": boxes,
                "labels": class_ids,
                "masks": torch.stack(masks) if isinstance(masks, list) else masks,
            })

        # If no transforms, convert manually
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, target
        
    def _resolve_image_path(self, info):
        category = info.get("filename", "").split()[0]  # fallback if category is in path eg ~/Clean/Clean (2).jpg
        return os.path.join(self.dataset_dir, category, info["filename"])
        
    def _generate_instance_masks(self, info, height, width):
        polygons = info["regions"]
        num_instances = len(polygons)

        mask = np.zeros((height, width, num_instances), dtype=np.uint8)
        class_ids = []
        boxes = []

        for i, region in enumerate(polygons):
            shape = region["shape_attributes"]
            attrs = region.get("region_attributes", {})
            shape_type = shape.get("name")
            class_name = attrs.get("class", "Clean").lower()   # the key here is type not class 
            class_id = self._map_class_name(class_name)
            class_ids.append(class_id)

            
            ## Check If I need to add another mask shape
            # Create mask
            if shape_type == "polygon":
                x = shape.get("all_points_x")
                y = shape.get("all_points_y")
                rr, cc = skimage.draw.polygon(y, x)
            elif shape_type == "rect":
                x, y = shape["x"], shape["y"]
                h, w = shape["height"], shape["width"]
                rr, cc = skimage.draw.rectangle(start=(y, x), extent=(h, w))   # row and column coordinates
            elif shape_type == "polyline":
                x = shape.get('all_points_x')
                y = shape.get('all_points_y')
                rr, cc = skimage.draw.polygon(y,x)
            else:
                continue  # skip unsupported types

            rr = np.clip(np.round(rr).astype(int), 0, height - 1)
            cc = np.clip(np.round(cc).astype(int), 0, width - 1)
            mask[rr, cc, i] = 1   # highlight the region of interest

            pos = np.where(mask[:, :, i])
            ymin, ymax = pos[0].min(), pos[0].max()
            xmin, xmax = pos[1].min(), pos[1].max()
            boxes.append([xmin, ymin, xmax, ymax])

        masks = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.uint8)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)    # unnecessary commend out 
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return masks, class_ids, boxes
    
    
    def _map_class_name(self, name):
        class_map = {
            "clean": 1,
            "dust": 2,
            "physical": 3,
            "electrical": 4,
            "bird": 5,
            "snow": 6
        }
        return class_map.get(name.lower(), 1)
    
    @staticmethod
    def _get_albumentations_transforms(train=True):
        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0),   # Normalize the image pixels to the [0, 1]
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            #mask_params=A.MaskParams())
        else:
            return A.Compose([ToTensorV2()]) 
        
        
        
IMAGE_DATA_DIR = 'c:/Users/panos/CVision/Data'
ANNOTATION_JSON_PATH = 'c:/Users/panos/CVision/Data/via_project_10Jul2025_15h51m_json.json'


class SolarConfig():
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "solar"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + Solar Panel Damage Categories

    # Numbe of training steps pre epoch
    STEPS_PER_EPOCH = 10

    # Skip detection with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


    USE_MINI_MASK = False
    
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")   # Here we load the weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
    
def evaluate(model, dataset_val, device):
    model.train()
    data_loader = DataLoader(dataset_val,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss
    

max_lr = 6e-3 
min_lr = max_lr * 0.1
warmup_steps = 10    
num_epochs = 30 
 
def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / (warmup_steps+1)
        # 2) in between, use cosine decay down to min learning rate
        #decay_ratio = (it - warmup_steps) / (num_epochs - warmup_steps)
        # Clamp decay_ratio to [0, 1] to prevent assertion errors in case of misaligned inputs
        decay_ratio = min(1.0, max(0.0, (it - warmup_steps) / (num_epochs - warmup_steps)))
        assert 0 <= decay_ratio <= 1 
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
        import matplotlib.pyplot as plt
        lrs = [get_lr(i) for i in range(num_epochs)]
        plt.plot(lrs); plt.title("Learning Rate Schedule"); plt.show()
        return min_lr + coeff * (max_lr - min_lr)
    
    

def train(model, dataset_train, dataset_val, device):
    data_loader = DataLoader(dataset_train,
                             batch_size=2,
                             shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))
    
    for param in model.backbone.parameters():
        param.requires_grad = False   # Freeze backbone

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        best_avg_val_loss = float('inf')
        # loop through our batch 
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()        
            optimizer.step()
            
            running_loss += losses.item()
        # Update the learning rate per epoch
        for param_group in optimizer.param_groups: 
            param_group['lr'] = get_lr(epoch)
            
        avg_train_loss = running_loss / len(data_loader)
        avg_val_loss = evaluate(model, dataset_val, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        # log to wandb
        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        })
        
        # Save the best model 
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_train_loss
            checkpoint_path = os.path.join(args.logs, f"best_model_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            # Log model to wandb
            wandb.save(checkpoint_path)
            print(f"Saved best model at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")


        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}")    
    

def color_splash(image, mask):
    # If no masks detected, return grayscale image
    if mask.size == 0:
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        return gray.astype(np.uint8)

    # Collapse all masks into one
    mask = (np.sum(mask, axis=0, keepdims=True) >= 1)  # [1, H, W]
    mask = mask.transpose(1, 2, 0)  # [H, W, 1]
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    splash = np.where(mask, image, gray).astype(np.uint8)
    return splash


def draw_boxes_on_splash(splash_image, output, threshold=0.8, class_names=None):
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()

    image_draw = splash_image.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)            
        label_text = f"{class_names[label] if class_names else label}: {score:.2f}"
        cv2.rectangle(image_draw, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.putText(image_draw, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

    return image_draw

def detect_and_color_splash_pytorch(model, image_path, device, threshold=0.8):
    model.eval()
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        output = model([image_tensor])[0]

    # Filter out low-confidence detections
    keep = output['scores'] > threshold
    masks = output['masks'][keep].squeeze(1)  # [N, H, W]

    # Convert image to numpy array
    image_np = np.array(image)
    print(f"Detections: {len(output['scores'])}, Above threshold: {keep.sum().item()}")
    # Create color splash
    splash = color_splash(image_np, masks.cpu().numpy())
    final_image = draw_boxes_on_splash(splash, output, threshold)

    file_name = "/content/drive/MyDrive/CVision/splash_with_boxes_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    cv2.imwrite(file_name, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print("Saved with boxes and splash:", file_name)

sys.argv = ['script.py', '--weights', 'C:/Users/panos/CVision/external/Mask_RCNN/mask_rcnn_coco.h5']


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 1 + 6  # background + 6 solar damage classes
model = get_model(num_classes)
model.to(device)


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
                    metavar=IMAGE_DATA_DIR,
                    help='Root directory of our dataset',
                    default=IMAGE_DATA_DIR
                    )
### THE FOLLOWING IS SHOULD BE COMMEND OUT ###  
parser.add_argument('--weights',
                    required=False,
                    help='Path to weights .pth file or "coco" ',
                    default=''
                    )

parser.add_argument('--logs',
                    required=False,
                    metavar=r'C:\Users\panos\CVision\Logs',
                    help='Path to logs and checkpoints',
                    default='C:/Users/panos/CVision/Logs'
                    )

parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

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
    dataset_train = SolarDataset(dataset_dir=IMAGE_DATA_DIR, annotation_dir=ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=True), mode="train", val_size=0.2)
    dataset_val = SolarDataset(dataset_dir=IMAGE_DATA_DIR, annotation_dir=ANNOTATION_JSON_PATH, transforms=None, mode="val", val_size=0.2)

    wandb.init(project='SolarPanel-Damage-Detector',
                name=f"run_{time.strftime('%Y%m%d-%H%M%S')}",
                # config={
                #     'epochs': num_epochs,
                #     'batch_size': data,
                #     'learning_rate': config.,
                #     'optimizer': 'AdamW'
                # }
                )
    wandb.watch(model, log="gradients", log_freq=30)

    train(model, dataset_train, dataset_val, device)

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