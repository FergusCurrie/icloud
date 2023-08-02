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
import fiftyone as fo
import os
import tempfile
from src.config import IcloudConfig
from src.logger import get_logger
from skimage import io


config = IcloudConfig()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def check_sample_in_tfrecord(sample: fo.Sample) -> bool:
    flag = sample["tfrecord"]
    if flag is None:
        return False
    assert type(flag) == bool
    return flag


def resize_and_save(file, file_new):
    img = io.imread(file)
    img = img[:500, :500]
    io.imsave(file_new, img)


def encode_dataset(
    image_filenames: list[str],
    labels: list[int],
    save_path: Path = config.TFRECORD_FILENAME,
):
    logger = get_logger()
    logger.debug("Encoding dataset")

    temp_dir = tempfile.TemporaryDirectory()
    try:
        os.remove(str(save_path))
    except:
        logger.debug("No record to remove")

    with tf.io.TFRecordWriter(str(save_path)) as writer:
        count = 0
        for image_path in image_filenames:
            image_new_path = temp_dir.name + "/" + image_path.split("/")[-1]
            resize_and_save(image_path, image_new_path)
            try:
                raw_file = tf.io.read_file(image_new_path)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image_raw": _bytes_feature(raw_file.numpy()),
                            "label": _int64_feature(labels),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                count += 1
            except:
                logger.debug(f"Failed to encode {image_path}")
                continue

        logger.debug(f"Encoded {count} images")


def encode_labelled_dataset():
    """
    Take the labelled dataset, and encode it into tfrecord format.
    """
    path = Path("/home/fergus/repos/icloud")
    dataset = fo.load_dataset("icloud")
    # labelled dataset
    labeled_view = dataset.match({"('new_field',)": {"$exists": True}})
    image_filenames = [str(x.filepath) for x in labeled_view.iter_samples()]
    labels = []
    for image_path in image_filenames:
        image_labels = [
            x.label for x in labeled_view[image_path]["('new_field',)"].classifications
        ]
        fergus_labels = 1 if "fergus" in image_labels else 0
        labels.append(fergus_labels)
    encode_dataset(image_filenames, labels)


if __name__ == "__main__":
    encode_labelled_dataset()
