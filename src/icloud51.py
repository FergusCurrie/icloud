"""
Dataset management. 
"""

import argparse
from pathlib import Path
import fiftyone as fo
from config import IcloudConfig
from dotenv import load_dotenv
import os

load_dotenv(".env")
config = IcloudConfig()
print(os.environ.get("FIFTYONE_CVAT_USERNAME"))


def setup():
    # The directory containing the dataset to import
    dataset_dir = str(config.PATH)

    # The type of the dataset being imported
    dataset_type = fo.types.ImageDirectory

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        name=config.FIFTYONE_DATASET_NAME,
    )
    dataset.persistent = True


def server():
    # Launch the App
    session = fo.launch_app()

    # Blocks execution until the App is closed
    session.wait()


def delete():
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    dataset.delete()


def stats():
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    fo.pprint(dataset.stats(include_media=True))


def save_all_labels():
    """
    load_annotations wukk download and merge annotations from cvat into 51.
    It will not delete the annotation task from 51.
    """
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    anno_key = "label_all"
    dataset.load_annotations(
        anno_key,
        url="http://localhost:8080",
    )


def label_all():
    """
    Creates cvat labelling job for all images in dataset.

    see: https://docs.voxel51.com/integrations/cvat.html#cvat-examples
    requires cvat server to be running
    """
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    anno_key = "label_all"
    try:
        print("cleared previous label run")
        dataset.delete_annotation_run(anno_key)
    except:
        pass
    # view = dataset.exists("metadata", True).limit(100)
    #

    view = dataset.match({"('new_field',)": {"$exists": False}}).limit(100)
    print(view)

    label_field = (
        "new_field",
    )  # a string indicating a new or existing label field to annotate
    label_type = "classifications"
    classes = list(config.LABELS)

    view.annotate(
        anno_key,
        backend="cvat",
        url="http://localhost:8080",
        label_field=label_field,
        label_type=label_type,
        classes=classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine which 51 process to run")
    parser.add_argument(
        "process", type=str, nargs=1, help="setting for 51 process type"
    )
    args = parser.parse_args()
    if args.process == ["setup"]:
        setup()
    elif args.process == ["server"]:
        server()
    elif args.process == ["delete"]:
        delete()
    elif args.process == ["stats"]:
        stats()
    elif args.process == ["label_all"]:
        label_all()
    elif args.process == ["save_all_labels"]:
        save_all_labels()
    else:
        raise Exception("Invalid 51 process type")
