"""
File containing tensorflow transform code for ml pipeline. 

most useful repo so far 
https://github.com/T-Sumida/tfx_image_classify/blob/main/pipeline/keras_utils.py
"""

import tensorflow as tf


# # Create a dictionary describing the features. This
image_feature_description = {
    "image_raw": tf.io.FixedLenFeature([], tf.string),
    "filename": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}

IMG_SIZE = 224


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def preprocess_image(image_bytes):
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    return image


def preprocessing_fn(inputs):
    """
    recieves a batch of inputs as a python dictionary
    key = name of feature
    """
    # decoded = tf.io.decode_base64(inputs["image_raw"])
    images = tf.map_fn(
        lambda x: tf.io.decode_jpeg(x[0], channels=3),
        inputs["image_raw"],
        dtype=tf.uint8,
    )
    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])

    return {
        "input_1": images,
        "labels_xf": inputs["label"],
    }  # potentially sitll bytes
