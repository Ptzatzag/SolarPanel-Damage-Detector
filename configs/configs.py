from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path 

@dataclass(slots=True)
class SolarConfig():
    """
    Base configuration for solar panel damage detection
    
    """
    # Give the configuration a recognizable name
    name: str = "solar"
    
    ROOT_DIR = Path(__file__).resolve().parents[1]
    
    # Paths and class names 
    class_names: list = ['Clean', 'Snow']#,'Dust', 'Physical Damage', 'Electrical Damage', 'Bird Drop', ]  
    IMAGE_DATA_DIR = '/Data'
    ANNOTATION_JSON_PATH = '/Data/SnowCOCO.json'
    # WEIGHTS = '/Logs/best_model_25.pth'
    LOGS = '/Logs'
    IMAGE_INF_EXAMPLE = '/Data/Physical/Physical (64).jpg'
    
    # Training schedule
    warmup_steps: int = 10
    max_lr: float = 4e-5
    min_lr: float = 4e-6
    num_epochs: int = 200

    # Data / model
    num_classes: int = 3  # background + 2 damage classes
    use_mini_mask: bool = False

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Hardware
    gpu_count: int = 1
    images_per_gpu: int = 1
    
    @property
    def batch_size(self) -> int:
        return self.gpu_count * self.images_per_gpu

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2 (background + 1 class).")
        if self.max_lr <= 0 or self.min_lr <= 0 or self.learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr cannot be greater than max_lr.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.gpu_count <= 0 or self.images_per_gpu <= 0:
            raise ValueError("gpu_count and images_per_gpu must be positive integers.")

    def display(self) -> None:
        print("\nConfigurations:")
        for key, value in self.to_dict().items():
            print(f"{key:30} {value}")
        print(f"{'batch_size':30} {self.batch_size}")
        print()


@dataclass(slots=True)
class InferenceConfig(SolarConfig):
    """Configuration for inference."""

    gpu_count: int = 1
    images_per_gpu: int = 1
    num_epochs: int = 1  # not used in inference, but kept for compatibility