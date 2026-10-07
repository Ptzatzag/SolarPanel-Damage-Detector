import json

import pytest
from PIL import Image

from dataset.dataset import SolarDataset


def test_dataset_target_lengths_match(synthetic_dataset):
    dataset = synthetic_dataset

    assert len(dataset) > 0
    for index in range(len(dataset)):
        _, target = dataset[index]
        number_of_objects = len(target["boxes"])
        for field in ("labels", "masks", "area", "iscrowd"):
            assert len(target[field]) == number_of_objects, (
                f"{field} count differs from boxes at dataset index {index}"
            )


@pytest.fixture(params=[1, 2])
def synthetic_dataset(tmp_path, request):
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
            "id": annotation_id, "image_id": 1, "category_id": 1,
            "bbox": [2, 2, 8, 8], "area": 64,
            "iscrowd": 0,
            "segmentation": [[2, 2, 10, 2, 10, 10, 2, 10]],
        } for annotation_id in range(1, request.param + 1)],
    }), encoding="utf-8")
    return SolarDataset(
        dataset_dir=tmp_path,
        annotation_path=annotation_path,
        transforms=None,
        mode="all",
    )


def test_empty_segmentation_is_rejected(synthetic_dataset):
    synthetic_dataset.map_imgID_to_annotations[1][0]["segmentation"] = []
    with pytest.raises(ValueError, match="segmentation"):
        synthetic_dataset[0]
