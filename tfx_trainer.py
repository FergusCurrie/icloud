import tensorflow as tf
from tensorflow.keras.applications.convnext import ConvNeXtTiny
import tfx
from tfx_bsl.public import tfxio
import fiftyone as fo
from pathlib import Path


def _input_fn():
    dataset = fo.load_dataset("icloud")
    labeled_view = dataset.match({"('new_field',)": {"$exists": True}})
    images, labels = [], []
    for img in labeled_view:
        # print(img)
        filepath: str = img.filepath
        a_labels = []
        for classification in img["('new_field',)"].classifications:
            a_labels.append(classification.label)
        image = tf.io.decode_jpeg(tf.io.read_file(filepath))
        images.append(image)
        labels.append(a_labels)
    # resizing
    images = [
        tf.image.resize(
            image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        for image in images
    ]
    fergus_labels = [1 if "fergus" in label else 0 for label in labels]
    dataset = tf.data.Dataset.from_tensor_slices((images, fergus_labels))
    return dataset


def _build_keral_model():
    model = tf.keras.applications.ConvNeXtTiny(
        model_name="convnext_tiny",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    inputs = model.inputs
    x = model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.softmax)(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)
    finetune_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return finetune_model


def run_fn(fn_args: tfx.components.FnArgs):
    """
    This may be only strictly neccassry function.
    """
    training_dataset = _input_fn()
    finetune_model = _build_keral_model()
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy()
    )
    history = finetune_model.fit(training_dataset.batch(16), epochs=10)
    finetune_model.save(fn_args.serving_model_dir, save_format="tf")
