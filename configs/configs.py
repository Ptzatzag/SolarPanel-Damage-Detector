from dataclasses import dataclass, asdict, field
from email.policy import default
from typing import Dict, Any
from pathlib import Path 
from huggingface_hub import hf_hub_download

@dataclass(slots=True)
class SolarConfig():
    """
    Base configuration for solar panel damage detection
    
    """
    root_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    image_data_dir: Path = field(init=False)
    annotation_json_path: Path = field(init=False)
    logs_dir: Path = field(init=False)
    weights_path: Path = field(init=False)
    inference_image_path: Path = field(init=False)

    # Classes
    class_names: list[str] = field(
        default_factory=lambda: ["Clean", "Snow"]
    )

    # Training schedule
    warmup_steps: int = 10
    max_lr: float = 4e-5
    min_lr: float = 4e-6
    num_epochs: int = 200

    # Data / model
    num_classes: int = 1 + 2  # background + classes [clean, snow], we have annotations for clean and snow 
    use_mini_mask: bool = False

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Hardware
    gpu_count: int = 1
    images_per_gpu: int = 1

    def __post_init__(self) -> None:
        self.image_data_dir = self.root_dir / "Data"
        self.annotation_json_path = self.image_data_dir / "Snow_Updated.json"   
        self.logs_dir = self.root_dir / "Logs"
        self.weights_path = self.logs_dir / "best_model_25.pth"
        self.inference_image_path = (
            self.image_data_dir / "Physical" / "Physical (64).jpg"
    )  
        
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

    def download_weights(self):
        weight_path = hf_hub_download(
            repo_id="Ptzatzag/solar-panel-detector",
            filename="best_modelmult.pth"
            )
        return Path(weight_path)

@dataclass(slots=True)
class InferenceConfig(SolarConfig):
    """Configuration for inference."""

    gpu_count: int = 1
    images_per_gpu: int = 1
    num_epochs: int = 1  # not used in inference, but kept for compatibility