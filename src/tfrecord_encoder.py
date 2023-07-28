"""
Convert jpg images into tfrecord format. Updates 51 for each sample, to store metadata 
confirming the image has been converted, and is in tfrecord.  

- The rule of thumb is to have at least 10 times as many files as there will be hosts reading data.
- Ideally, you should shard the data to ~10N files, as long as ~X/(10_N) 
is 10+ MBs (and ideally 100+ MBs, X is size of dataset in GB).

Currently 3gb, 1 host. 
"""

import tensorflow as tf
from pathlib import Path
from logger import get_logger
import fiftyone as fo

from config import IcloudConfig
from logger import get_logger

config = IcloudConfig()
logger = get_logger()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes__for_string_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
    )


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


def encode_dataset():
    logger = get_logger()
    logger.debug("Encoding dataset")
    dataset = fo.load_dataset(config.FIFTYONE_DATASET_NAME)
    filenames: str = [
        str(x)
        for x in (config.ICLOUD_DATA_PATH / "raw_icloud").iterdir()
        if x.suffix == ".jpg" and not check_sample_in_tfrecord(dataset[str(x)])
    ]

    with tf.io.TFRecordWriter(str(config.TFRECORD_FILENAME)) as writer:
        count = 0
        for image_path in filenames:
            try:  # TODO: write with context handler
                raw_file = tf.io.read_file(image_path)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image_raw": _bytes_feature(raw_file.numpy()),
                            "filename": _bytes__for_string_feature(image_path),
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
    encode_dataset()
