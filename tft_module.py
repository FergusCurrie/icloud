"""
File containing tensorflow transform code for ml pipeline. 
"""

import tensorflow as tf


def preprocessing_fn(inputs):
    """
    recieves a batch of inputs as a python dictionary
    key = name of feature
    """
    assert type(inputs) == dict
    # print(f"\n\n\n\n\n {inputs['label']} \n\n\n\n\n") . shpae of (None, 1)
    # print(f"\n\n\n\n\n {inputs['image_raw']} \n\n\n\n\n")  # . shpae of (None, 1)
    raw_image = inputs["image_raw"]
    # raw_image = tf.reshape(raw_image, [-1])
    # raw_image = tf.io.decode_jpeg(raw_image)
    # logger.debug(f"rawimg shpa= {raw_image.shape}")
    # raw_image = tf.reshape(raw_image, [-1])
    # image = raw_image.map(tf.io.decode_jpeg)
    # image = (raw_image, channels=3)
    return {
        "input_1": raw_image,
        "labels_xf": inputs["label"],
    }  # potentially sitll bytes
