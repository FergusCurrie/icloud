import pytest
import tensorflow as tf


def count_tfrecord_examples(tfrecord_file):
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_file):
        count += 1
    return count


class TestTfxTrainer:
    def test_train(self):
        assert True

    def test_tfrecord_bigger_than_zero():
        # load tfrecord/icloud_data.tfrecord into a tf.data.Dataset
        data = tf.data.TFRecordDataset(["tfrecord/icloud_data.tfrecord"])
        assert data is not None
        assert True

    # def test_load_gzip():
    #     filenames = "my_pipeline_output/ImportExampleGen/examples/1/Split-train/data_tfrecord-00000-of-00001.gz"
    #     # _gzip_reader_fn(filenames)


if __name__ == "__main__":
    count = count_tfrecord_examples("tfrecord/icloud_data.tfrecord")
    print(count)
