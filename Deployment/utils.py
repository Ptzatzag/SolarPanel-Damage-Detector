import torch
import torchvision.transforms as T
from PIL import Image
import io
from torchvision.models.detection import maskrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
 
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")   # Here we load the weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_model(model_path, device):
    num_classes = 1 + 1  # background + 1 solar damage classes
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")   # Here we load the weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # [1, C, H, W]