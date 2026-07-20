import torch 
import numpy as np
import torchvision.transforms as T
from PIL import Image
import skimage.io
import skimage.draw
from torch.utils.data import DataLoader
import torchvision.transforms as T
import math
import cv2
import datetime
from configs.configs import SolarConfig

config = SolarConfig()  

def calc_validation_loss(model, dataset_val, device):
    model.train()   # Mask RCNN returns list of detections in the eval mode, we need loss dict

    # Hack for simulating eval mode, by switching Batch norm and dropout layers to eval mode
    for module in model.modules():
      if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval()
      if isinstance(module, torch.nn.modules.Dropout):
        module.eval()
    
    data_loader = DataLoader(dataset_val,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loss = 0.0
    # with torch.no_grad():   # issues with no tracking the gradient while being on train mode
    with torch.set_grad_enabled(False):
      for images, targets in data_loader:
          images = [img.to(device) for img in images]
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          #########
          with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
          #print(f"loss in the eval: {losses.item()}")
          val_loss += losses.item()
      # cleanup to avoid memory accumulation
      del loss_dict, losses, images, targets
      torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss
        
        
def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_steps:
            return config.max_lr * (it+1) / (config.warmup_steps+1)
        # 2) in between, use cosine decay down to min learning rate
        # Clamp decay_ratio to [0, 1] to prevent assertion errors in case of misaligned inputs
        decay_ratio = min(1.0, max(0.0, (it - config.warmup_steps) / (config.num_epochs - config.warmup_steps)))
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
        return config.min_lr + coeff * (config.max_lr - config.min_lr)

    
    
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


def draw_boxes_on_splash(splash_image, output, threshold=0.7, class_names=config.class_names):
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

def detect_and_color_splash_pytorch(model, image_path, device, threshold=0.7):
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
    final_image = draw_boxes_on_splash(splash, output, threshold, config.class_names)

    file_name = "/content/drive/MyDrive/CVision/splash_with_boxes_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    cv2.imwrite(file_name, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print("Saved with boxes and splash:", file_name)
