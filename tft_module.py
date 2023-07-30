"""
File containing tensorflow transform code for ml pipeline. 
"""

import tensorflow as tf


def preprocessing_fn(inputs):
    raw_image = inputs["image_raw"]
    img_rgb = tf.io.decode_jpeg(raw_image, channels=3)
    resized_img = tf.image.resize_with_pad(img_rgb, target_height=224, target_width=224)
    return {"image": resized_img}
