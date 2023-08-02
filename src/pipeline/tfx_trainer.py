import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio
import tfx


LABEL_KEY = "labels"


def transformed_name(key):
    return key + "_xf"


def build_keral_model() -> tf.keras.Model:
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
    for layer in finetune_model.layers:
        layer.trainable = True
    return finetune_model


def input_fn(
    file_pattern,
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size=16,
):
    """
    Function to load from tfrecords in pipeline output.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key="labels_xf"),
        schema=tf_transform_output.transformed_metadata.schema,
    )


def run_fn(fn_args: tfx.v1.components.FnArgs) -> None:
    """
    Code to train and save trained model.

    Args:
        @fn_args: Holds args used to train the model as name/value pairs.
    """
    # First load datasets which have been saved to tfx metadata layer.
    # tf_transform_output is a wrapper around the output of tf.Transform.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    # Build, traing and save.
    finetune_model = build_keral_model()
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy()
    )
    history = finetune_model.fit(train_dataset, epochs=10, steps_per_epoch=100)
    finetune_model.save(
        fn_args.serving_model_dir, save_format="tf"
    )  # TODO: serving signature
