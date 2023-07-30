import tensorflow as tf
from tensorflow.keras.applications.convnext import ConvNeXtTiny
import tfx


def _input_fn() -> tf.data.Dataset:
    pass


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
