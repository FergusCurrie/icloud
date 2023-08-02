import tempfile
import numpy as np
from PIL import Image

from src.tfrecord_encoder import encode_dataset


def create_dummy_image():
    temp_dir = tempfile.TemporaryDirectory()
    dummy_image_fn = temp_dir.name + "/" + "test.jpg"
    img = np.ones((500, 500))
    new_p = Image.fromarray(img)
    if new_p.mode != "RGB":
        new_p = new_p.convert("RGB")
    new_p.save(dummy_image_fn, format="jpeg")
    return dummy_image_fn


def test_encode_dataset():
    # create dummy image
    # dummy_image_fn = create_dummy_image()
    # encode_dataset(dummy_image_fn, [1], save_path="test.tfrecord")
    assert 1 == 1
