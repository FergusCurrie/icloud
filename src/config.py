from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()


@dataclass
class IcloudConfig:
    FIFTYONE_DATASET_NAME: str = "icloud"
    APPLE_ID: str = os.environ.get("APPLE_ID")
    APPLE_PASSWORD: str = os.environ.get("APPLE_PASSWORD")
    ICLOUD_DATA_PATH: Path = Path("/home/fergus/data/icloud_data/")
    TFRECORD_FILENAME: Path = ICLOUD_DATA_PATH / "tfrecord/icloud_data.tfrecord"
    LABELS: tuple[str] = (
        "screen_shot",
        "fergus",
        "contains_people",
        "contains_animals",
        "scenery",
        "contains_text",
        "contains_food",
        "my_fitness_pal",
        "can",
        "indoors",
    )
