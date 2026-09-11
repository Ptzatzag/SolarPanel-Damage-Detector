from pathlib import Path
from configs.configs import SolarConfig

def test_config_constructs_paths():
    config = SolarConfig(root_dir=Path("/project"))

    assert config.image_data_dir == Path("/project/Data") 
    assert config.annotation_json_path == Path(
        "/project/Data/Snow_Updated.json"
    )
    assert config.logs_dir == Path("/project/Logs")

def test_number_of_classes_includes_background():
    config = SolarConfig()

    assert config.num_classes == len(config.class_names) + 1