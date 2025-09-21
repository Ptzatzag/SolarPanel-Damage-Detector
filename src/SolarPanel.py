## ToDo
# trim the code with config
# change the splas functions 
##

import torch 
import numpy as np
import json
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
from torchvision import transforms
import math
import cv2
import datetime    
import sys
import argparse
import wandb
import time
from pycocotools import mask as pycoco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json 
import io


class_names = ['Clean', 'Dust', 'Physical Damage', 'Electrical Damage', 'Bird Drop', 'Snow', ]      
IMAGE_DATA_DIR = 'c:/Users/panos/CVision/Data'
ANNOTATION_JSON_PATH = 'c:/Users/panos/CVision/Data/via_project_10Jul2025_15h51m_json.json'



class SolarDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 annotation_dir: str,
                 transforms: Optional[Callable]=None,
                 mode: str = "train",
                 val_size: float = 0.2) -> None:
        self.dataset_dir = dataset_dir
        self.transforms = transforms

        with open(annotation_dir) as f:
            annotation_dict = json.load(f)

        # Extract information from COCO dict
        self.annotation_info = [img for img in annotation_dict.get("annotations")]   # keys: ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd', 'attributes']
        
        # The following inclues only one annotation per image
        # Lookup table of all annotations of each image, with keys: image_id and values: dict{annotations}
        self.image_id_to_info = {img['id']: img for img in annotation_dict.get("images")}


        # This should be the right one  
        self.map_imgID_to_annotations = {}
        for ann_info in self.annotation_info:
            key = ann_info.get('image_id')
            if key in self.map_imgID_to_annotations:
                self.map_imgID_to_annotations[key].append(ann_info)
            else:
                self.map_imgID_to_annotations[key] = [ann_info]


        # Filter out images with no annotations
        all_image_ids_with_annotations = list(self.map_imgID_to_annotations.keys())


        # Get all image IDs that have at least one annotation
        if mode in ["train", "val"]:
            train_ids, val_ids = train_test_split(all_image_ids_with_annotations, test_size=val_size, random_state=99)

            self.image_ids = train_ids if mode == "train" else val_ids
        else:
            self.image_ids = all_image_ids_with_annotations

        self.image_infos = [self.image_id_to_info[k] for k in self.image_ids]
        self.annotation_info = [self.map_imgID_to_annotations[k] for k in self.image_ids]    ### Check this

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        info = self.image_infos[idx]
        image_id = self.image_ids[idx]
        # Load image
        image_path = self._resolve_image_path(info)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        height, width = info['height'], info['width']


        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Process annotations for this image
        annotations = self.map_imgID_to_annotations.get(image_id, [])
       ############################################################## 
####### Run though annotations and get the boxes and masks for each key
        masks = []
        boxes = []
        boxes_xyxy = []
        labels = []
        iscrowd_flags = []
        annotations = self.map_imgID_to_annotations.get(image_id, [])

        for ann in annotations:
            # Convert COCO bbox (x, y, w, h) → xyxy
            x_min, y_min, w, h = ann['bbox']

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            x_max = x_min + w
            y_max = y_min + h

            # if w > 0 and h > 0:  # filter invalid boxes
            #     boxes_xyxy.append([x_min, y_min, x_max, y_max])
            # else:
            #     continue  # skip invalid


            # Clamp coordinates to image boundaries
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(x_min + 1, min(x_max, width))
            y_max = max(y_min + 1, min(y_max, height))
            
            boxes_xyxy.append([x_min, y_min, x_max, y_max])

            rle_mask = ann['segmentation']
            rles = pycoco_mask.frPyObjects(rle_mask, height, width)

            decoded_mask = pycoco_mask.decode(rles)
            if decoded_mask.ndim == 2:  # single mask
                masks.append(decoded_mask)
            else:  # multiple masks from the same annotation
                for i in range(decoded_mask.shape[-1]):
                    masks.append(decoded_mask[..., i])

            # masks.append(decoded_mask)
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            iscrowd_flags.append(ann['iscrowd'])


        # Convert to tensors
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.empty((0, 4), dtype=torch.float32)
        #boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        masks_tensor = torch.stack([torch.from_numpy(m).to(torch.uint8) for m in masks]) if masks else torch.empty((0, height, width), dtype=torch.uint8)
        iscrowd_tensor = torch.tensor(iscrowd_flags, dtype=torch.int64) if iscrowd_flags else torch.empty((0,), dtype=torch.int64)
        # Calculate area for COCO (bbox = [x, y, w, h])
       # area_tensor = boxes_tensor[:, 2] * boxes_tensor[:, 3] if boxes_tensor.numel() > 0 else torch.tensor([], dtype=torch.float32)

        # Correct area calculation (now in xyxy format)
        if boxes_tensor.numel() > 0:
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            area_tensor = torch.tensor([], dtype=torch.float32)


        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([image_id]),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor#self.annot_ids[idx].get("iscrowd")
        }

        if self.transforms:
          # Albumentations expects NumPy arrays for image, masks, and lists for bboxes/labels
          # It returns a dictionary, and the outputs are typically already tensors if ToTensorV2 is used.
          transformed = self.transforms(
              image=image,
              masks=masks_tensor.cpu().numpy() if masks_tensor.numel() > 0 else np.empty((0, image.shape[0], image.shape[1]), dtype=np.uint8),
              bboxes=boxes_tensor.cpu().tolist() if boxes_tensor.numel() > 0 else [],
              labels=labels_tensor.cpu().tolist() if labels_tensor.numel() > 0 else []
          )
          image = transformed["image"] # This will already be a tensor if ToTensorV2 is in transforms

          # Ensure transformed masks are correctly handled (could be tensor or list of tensors)
          if isinstance(transformed["masks"], list): # If transform returns list of masks (e.g. sometimes without ToTensorV2)
              target["masks"] = torch.stack(transformed["masks"]) if transformed["masks"] else torch.empty((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
          else: # If transform returns a single stacked tensor for masks
              target["masks"] = transformed["masks"] # This should already be a tensor

          # Convert transformed bboxes and labels back to tensors
          target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else torch.empty((0, 4), dtype=torch.float32)
          target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64) if transformed["labels"] else torch.empty((0,), dtype=torch.int64)

          # Recalculate area and iscrowd if bounding boxes changed during transform
          target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0]) if target["boxes"].numel() > 0 else torch.tensor([], dtype=torch.float32)
          target["iscrowd"] = torch.zeros((len(target["boxes"]),), dtype=torch.int64) # All transformed objects are considered non-crowd

        else:
          # If no transforms, manually convert image to tensor (C, H, W) and normalize
          image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target



        ##########################################
        ########## TRANSFORMATIONS HERE ##########
        ##########################################


        # image = torch.from_numpy(image).permute(2,0,1).float() / 255.0   # understand this
        # return image, target

    def _resolve_image_path(self, info):
        category = info.get("file_name", "").split()[0]
        return os.path.join(self.dataset_dir, category, info.get("file_name", ""))

    @staticmethod
    def _get_albumentations_transforms(train):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.5),
                # add more transformations 
                A.VerticalFlip(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=400, p=0.1),  # Sun glare
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),  # Shadows
                A.CoarseDropout(
                              max_holes=8,
                              hole_height_range=(0.05, 0.2),  # fraction of image height
                              hole_width_range=(0.05, 0.2),   # fraction of image width
                              p=0.2
                          ),  # Simulate dirt spots

                # Normalize using ImageNet's mean and std
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                # ToTensorV2 should generally be the last step,
                # converting NumPy arrays to PyTorch tensors and permuting dimensions (H, W, C) -> (C, H, W)
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc', # Ensure this matches your [xmin, ymin, xmax, ymax] format
                label_fields=['labels']
            )
           #mask_params=A.MaskParams() # <-- IMPORTANT: Uncomment and enable this for mask transformations
           )
        else:
            # For validation/inference, typically only normalization and tensor conversion are needed
            return A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc', 
                label_fields=['labels']
            ),
            #mask_params=A.MaskParams() # Also needed for validation if masks are part of output
            )
   

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
    
def calc_validation_loss(model, dataset_val, device):
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
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss



def evaluate(model, dataset_val, device, annotation_dir):
    # print(f"Evaluate step | Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
    #     f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    model.eval()
    data_loader = DataLoader(dataset_val,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    predictions = []


    image_ids_val = sorted(dataset_val.image_ids) # Check the attributes name
    #print("Validation image IDs: ", image_ids_val)
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
     #       print(f"Processing image {i+1} of {len(data_loader)}")
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
              outputs = model(images)
            # Move outputs to CPU
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

########### ADD AUTOCAST HERE
            # Process outputs and convert to COCO format
            for img_idx, output in enumerate(outputs):
                image_id = targets[img_idx]['image_id'].item()


                # Check if model predicted any instances
                if len(output['boxes']) == 0:
                    continue

                # Filter out predictions with low confidence scores (e.g., < 0.05 or 0.1)
                # This helps in mAP calculation by reducing many low-quality FPs
                score_threshold = 0.5 # You can tune this threshold
                keep = output['scores'] > score_threshold

                boxes = output['boxes'][keep].numpy()
                labels = output['labels'][keep].numpy()
                scores = output['scores'][keep].numpy()
                masks = output['masks'][keep].numpy()


                # Post-process masks: Threshold and convert to binary (if not already)
                # and ensure it's (N, H, W) binary mask if not already
                # A common threshold for Mask R-CNN outputs is 0.5
                masks = (masks > 0.5).astype(np.uint8)

                # Iterate through each detected instance
                for j in range(len(boxes)):
                    #print(f"every detected box: {boxes[j]}")
                    bbox = boxes[j]
                    label = labels[j]
                    score = scores[j]
                    mask = masks[j][0]# if masks[j].ndim == 4 else masks[j] # Remove channel dim if present

                    # Convert bounding box from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]
                    bbox_coco = [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2] - bbox[0]),
                        float(bbox[3] - bbox[1])
                    ]

                    #mask = (mask > 0.5).astype(np.uint8)
                    # Ensure mask is contiguous for RLE encoding
                    mask = np.asfortranarray(mask)
                    # Convert binary mask to RLE format
                    rle = pycoco_mask.encode(mask)
                    # Convert RLE counts to string for JSON serialization
                    rle['counts'] = rle['counts'].decode('utf-8')
                  #  print("CATEGORY ID: ", int(label))
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label), # Ensure category_id is int
                        "bbox": bbox_coco,
                        "score": float(score),
                        "segmentation": rle
                    })

    # --- COCO Evaluation ---
    # Load ground truth annotations
    coco_gt = COCO(annotation_dir)
    if not predictions:
        print("No predictions generated for evaluation. mAP will be 0.")
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
        return 0.0, 0.0 # Return 0 for both bbox and mask mAP if no predictions

    coco_dt = coco_gt.loadRes(predictions)

    # --- Evaluate bounding box detections ---
    # Redirect stdout to capture pycocotools print statements
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    print("\n--- Evaluating Bounding Boxes ---")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    # Filter to only evaluate images that were actually in the validation dataset
    coco_eval_bbox.params.imgIds = image_ids_val
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()   #  final summary statistics
    # Extract mAP (IoU=.50:.05:.95) for bounding boxes
    mAP_bbox = coco_eval_bbox.stats[0]

    print("\n--- Evaluating Masks ---")
    coco_eval_mask = COCOeval(coco_gt, coco_dt, 'segm') # Use 'segm' for mask evaluation
    coco_eval_mask.params.imgIds = image_ids_val
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()
    # Extract mAP (IoU=.50:.05:.95) for masks
    mAP_mask = coco_eval_mask.stats[0]

    # Restore stdout
    captured_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    #print(captured_output) # Print captured pycocotools output
    torch.cuda.empty_cache()

    return mAP_bbox, mAP_mask
    

max_lr = 6e-3 
min_lr = max_lr * 0.1
warmup_steps = 10    
num_epochs = 30 
 
def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / (warmup_steps+1)
        # Clamp decay_ratio to [0, 1] 
        decay_ratio = min(1.0, max(0.0, (it - warmup_steps) / (num_epochs - warmup_steps)))
        assert 0 <= decay_ratio <= 1 
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    
    

def train(model, dataset_train, dataset_val, device, unfreeze_epoch):
    data_loader = DataLoader(dataset_train,
                             batch_size=1,
                             shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))


    for param in model.backbone.body.parameters():   # freeze only resnet body
      param.requires_grad = False


    for param in model.backbone.fpn.parameters():   # keep FPN trainable, better for performance
        param.requires_grad = True


    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_avg_val_loss = float('inf')
    patience = 50
    epochs_no_improve = 0
    checkpoint_path = os.path.join(args.logs, f"best_model.pth")
#################### NO IDEA ABOUT SCALER ####################
    scaler = torch.amp.GradScaler('cuda')    # Gradient scaler, because we use low percision float16 and the grad could underflow
#################### NO IDEA ABOUT SCALER ####################

    accumulation_steps = 16  # effective batch size
    for epoch in range(num_epochs):
        # print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
        # f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        if epoch == unfreeze_epoch:
          print(f"Finetune, by unfreeze leyer 4 at epoch {epoch}")
          # for param in model.backbone.parameters():
          #     param.requires_grad = True
          for name, param in model.backbone.named_parameters():
            if "layer4" in name:
              param.requires_grad = True
        elif epoch == unfreeze_epoch + 50:
          print(f"Activate Layer 3 at epoch {epoch}")
          for name, param in model.backbone.named_parameters():
            if "layer3" in name:
              param.requires_grad = True
        elif epoch == unfreeze_epoch + 70:
          print(f"Activate Layer 2 at epoch {epoch}")
          for name, param in model.backbone.named_parameters():
            if "layer2" in name:
              param.requires_grad = True
        elif epoch == unfreeze_epoch + 90:
          print(f"Active Layer 1 at epoch {epoch}")
          for name, param in model.backbone.named_parameters():
            param.requires_grad = True

        model.train()
        running_loss = 0.0


        # Run through our batch
        for step, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Use Automatic Mixed Precision (AMP) to reduce the overhead
            with torch.autocast(device_type='cuda', dtype=torch.float16):
              loss_dict = model(images, targets)
              loss = sum(loss for loss in loss_dict.values())
              loss = loss / accumulation_steps   # scale for accumulation

            scaler.scale(loss).backward()   # scale the loss
            #optimizer.zero_grad()
            #losses.backward()
            #optimizer.step()
            if (step + 1) % (accumulation_steps) == 0:
              scaler.step(optimizer)    # unscale the gradients before update
              scaler.update()           # update the scale for the next iteration
              optimizer.zero_grad()


            running_loss += loss.item() * accumulation_steps

        avg_train_loss = running_loss / len(data_loader)
        # Clean up memory
        torch.cuda.empty_cache()
        avg_val_loss = calc_validation_loss(model, dataset_val, device)

        # # Update the learning rate per epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)

        current_lr = optimizer.param_groups[0]['lr']

        log_data = {
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr,
        }

        # Evaluate 3rd epoch
        if (epoch + 1) % 3 == 0:
            mAP_bbox, mAP_mask = evaluate(model, dataset_val, device, ANNOTATION_JSON_PATH)
            log_data['mAP_bbox'] = mAP_bbox
            log_data['mAP_mask'] = mAP_mask

        # Log everything for the current epoch in a single call
        wandb.log(log_data)

        # Save the best model
        if avg_val_loss < best_avg_val_loss:
          best_avg_val_loss = avg_val_loss
          epochs_no_improve = 0
          torch.save(model.state_dict(), checkpoint_path)

          # checkpoint_path = os.path.join(args.logs, f"best_model_{epoch}.pth")
          # torch.save(model.state_dict(), checkpoint_path)
          #wandb.save(checkpoint_path)
          print(f"Saved best model at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")
        else:
          epochs_no_improve += 1

        # Add early stopping
        if epochs_no_improve > patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement")
            break

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


    # Log model to wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)


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
    final_image = draw_boxes_on_splash(splash, output, threshold, class_names=class_names)

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
    dataset_val = SolarDataset(dataset_dir=IMAGE_DATA_DIR, annotation_dir=ANNOTATION_JSON_PATH, transforms=SolarDataset._get_albumentations_transforms(train=False), mode="val", val_size=0.2)

    wandb.init(project='SolarPanel-Damage-Detector',
                name=f"run_{time.strftime('%Y%m%d-%H%M%S')}",
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