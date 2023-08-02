"""
File containing tensorflow transform code for ml pipeline. 

most useful repo so far 
https://github.com/T-Sumida/tfx_image_classify/blob/main/pipeline/keras_utils.py
"""

import tensorflow as tf

IMG_SIZE = 224


def preprocessing_fn(inputs: dict) -> dict:
    """
    Convert the tfrecord into a dictionary of tensors.

    The dictionary is dependent on how the TFRecord was
    built.

    Args:
        @inputs: a dictionary of feature to batch of byte inputs
    Returns:
        @outputs: a dictionary of feature to batch of tensors
    """
    outputs = {}
    images = tf.map_fn(
        lambda x: tf.io.decode_jpeg(x[0], channels=3),
        inputs["image_raw"],
        dtype=tf.uint8,
    )
    images = tf.cast(images, tf.float32)
    outputs["input_1"] = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    outputs["labels_xf"] = inputs["label"]
    return outputs
