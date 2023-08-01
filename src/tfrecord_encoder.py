"""
Convert jpg images into tfrecord format. Updates 51 for each sample, to store metadata 
confirming the image has been converted, and is in tfrecord.  

- The rule of thumb is to have at least 10 times as many files as there will be hosts reading data.
- Ideally, you should shard the data to ~10N files, as long as ~X/(10_N) 
is 10+ MBs (and ideally 100+ MBs, X is size of dataset in GB).

Currently 3gb, 1 host. 
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import fiftyone as fo
import os
import tempfile
from src.config import IcloudConfig
from src.logger import get_logger
from skimage import io


config = IcloudConfig()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes__for_string_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
    )


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def udpate_fiftyone_metadata(image_path: str, dataset: fo.Dataset) -> None:
    sample = dataset[image_path]
    sample["tfrecord"] = True
    sample.save()


def check_sample_in_tfrecord(sample: fo.Sample) -> bool:
    flag = sample["tfrecord"]
    if flag is None:
        return False
    assert type(flag) == bool
    return flag


def create_500_500_dataset(file, file_new):
    # create a temp directory

    img = io.imread(file)
    img = img[:500, :500]
    io.imsave(file_new, img)


def encode_dataset(dataset, save_path: Path = config.TFRECORD_FILENAME):
    logger = get_logger()

    logger.debug("Encoding dataset")

    # filenames: str = [
    #     str(x)
    #     for x in (config.ICLOUD_DATA_PATH / "raw_icloud").iterdir()
    #     if x.suffix == ".jpg"  # and not check_sample_in_tfrecord(dataset[str(x)])
    # ]
    filenames = [str(x.filepath) for x in dataset.iter_samples()]
    temp_dir = tempfile.TemporaryDirectory()
    # remove file with os.remove
    os.remove(save_path / "tfrecord/icloud_data.tfrecord")

    with tf.io.TFRecordWriter(
        str(save_path / "tfrecord/icloud_data.tfrecord")
    ) as writer:
        count = 0
        for image_path in filenames:
            try:  # TODO: write with context handler
                # print(image_path.split("/"))
                # print(temp_dir.name)
                image_new_path = temp_dir.name + "/" + image_path.split("/")[-1]
                # print(image_new_path)

                create_500_500_dataset(image_path, image_new_path)

                raw_file = tf.io.read_file(image_new_path)
                labels = [
                    x.label
                    for x in dataset[image_path]["('new_field',)"].classifications
                ]
                fergus_labels = 1 if "fergus" in labels else 0
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image_raw": _bytes_feature(raw_file.numpy()),
                            # "filename": _bytes__for_string_feature(image_path),
                            "label": _int64_feature(fergus_labels),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                udpate_fiftyone_metadata(image_path, dataset)
                logger.debug(f"Encoded {image_path} images")
                count += 1
            except FileNotFoundError:
                print(f"File {image_path} could not be found")
                continue
        logger.debug(f"Encoded {count} images")


if __name__ == "__main__":
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    encode_dataset(dataset)
