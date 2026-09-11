import json

import pytest
from PIL import Image

from configs.configs import SolarConfig
from dataset.dataset import SolarDataset


def test_dataset_target_lengths_match():
    config = SolarConfig()

    dataset = SolarDataset(
        dataset_dir=config.image_data_dir,
        annotation_path=config.annotation_json_path,
        transforms=None,
        mode="all",
        category_mapping=None
    )

    assert len(dataset) > 0
    for index in range(len(dataset)):
        _, target = dataset[index]
        number_of_objects = len(target["boxes"])
        for field in ("labels", "masks", "area", "iscrowd"):
            assert len(target[field]) == number_of_objects, (
                f"{field} count differs from boxes at dataset index {index}"
            )


@pytest.fixture
def dataset_with_empty_mask(tmp_path):
    image_dir = tmp_path / "Snow"
    image_dir.mkdir()
    Image.new("RGB", (16, 16)).save(image_dir / "Snow (1).png")
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({
        "images": [{
            "id": 1, "file_name": "Snow (1).png", "height": 16, "width": 16,
        }],
        "categories": [{"id": 1, "name": "Snow"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [2, 2, 8, 8], "area": 64,
            "iscrowd": 0, "segmentation": [],
        }],
    }), encoding="utf-8")
    return SolarDataset(
        dataset_dir=tmp_path,
        annotation_path=annotation_path,
        transforms=None,
        mode="all",
    )


def test_empty_segmentation_is_rejected(dataset_with_empty_mask):
    with pytest.raises(ValueError, match="segmentation"):
        dataset_with_empty_mask[0]
