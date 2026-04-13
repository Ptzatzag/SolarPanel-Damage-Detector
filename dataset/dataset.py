import torch 
import numpy as np
import json
import os 
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Optional, Callable
from torch.utils.data import Dataset
from pycocotools import mask as pycoco_mask
import json 
import albumentations as A
from albumentations.pytorch import ToTensorV2



class SolarDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 annotation_dir: str,
                 transforms: Optional[Callable]=None,
                 mode: str="train",
                 val_size: float = 0.2) -> None:
        self.dataset_dir = dataset_dir
        self.transforms = transforms

        with open(annotation_dir) as f:
            annotation_dict = json.load(f)

        self.annotation_info = [img for img in annotation_dict.get("annotations")]   # keys: ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd', 'attributes']

        # Build a map of image_id to image_info for efficient lookup
        self.image_id_to_info = {img['id']: img for img in annotation_dict.get("images")}


        self.map_imgID_to_annotations = {}
        for ann_info in self.annotation_info:
            # print(idx, ann_info)
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
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        masks_tensor = torch.stack([torch.from_numpy(m).to(torch.uint8) for m in masks]) if masks else torch.empty((0, height, width), dtype=torch.uint8)
        iscrowd_tensor = torch.tensor(iscrowd_flags, dtype=torch.int64) if iscrowd_flags else torch.empty((0,), dtype=torch.int64)

        # Correct area calculation (now in xyxy format)
        if boxes_tensor.numel() > 0:
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            area_tensor = torch.tensor([], dtype=torch.float32)


        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor(image_id),   # ([image_id])
            "area": area_tensor,
            "iscrowd": iscrowd_tensor
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
    
    def _resolve_image_path(self, info):
      category = info.get("file_name", "").split()[0]
      return os.path.join(self.dataset_dir, category, info.get("file_name", ""))

    @staticmethod
    def _get_albumentations_transforms(train):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        bbox_params_config = A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=0.0,
        min_visibility=0.0)

        if train:
          return A.Compose(
            [
                #A.Resize(height=800, width=800),   # Resize the image to lower the overhead
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),
                A.VerticalFlip(p=0.3), # Added this as it was in your snippet
                A.RandomBrightnessContrast(p=0.2),
        ####### Update the following transformations #######
                # --- Geometric Transformations ---
                #A.Rotate(limit=30, p=0.5), # Increased rotation limit
                # A.ShiftScaleRotate(
                #     shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.3
                # ), # Minor shifts and zooms
                #A.ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), # Distortions

                # --- Noise and Occlusions (important for robustness) ---
                #A.GaussNoise(var_limit=(10, 50), p=0.2), # Add Gaussian noise

                # Normalize image pixels using ImageNet's mean and std
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                # Converts image and masks to PyTorch tensors, and changes image format from HWC to CHW
                ToTensorV2()
            ], bbox_params=bbox_params_config)
        else:
            # For validation/inference, typically only normalization and tensor conversion are needed
            return A.Compose([
                #A.Resize(height=800, width=800),   # Resize the image to lower the overhead
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2()
                ],
                bbox_params=bbox_params_config)