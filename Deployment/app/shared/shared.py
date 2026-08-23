import torchvision.transforms as T
from PIL import Image


def preprocess_image(image: Image.Image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # [1, C, H, W]