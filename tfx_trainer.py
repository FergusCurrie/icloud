import tensorflow as tf

# from tensorflow.keras.applications.convnext import ConvNeXtTiny
import tfx

# import tensorflow transform
import tensorflow_transform as tft
from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
from tfx_bsl.public import tfxio
from tfx.utils import logging_utils

# import fiftyone as fo
# from pathlib import Path

LABEL_KEY = "labels"


# def _old_input_fn():
#     """
#     Function to manually load from raw.
#     """
#     dataset = fo.load_dataset("icloud")
#     labeled_view = dataset.match({"('new_field',)": {"$exists": True}})
#     images, labels = [], []
#     for img in labeled_view:
#         # print(img)
#         filepath: str = img.filepath
#         a_labels = []
#         for classification in img["('new_field',)"].classifications:
#             a_labels.append(classification.label)
#         image = tf.io.decode_jpeg(tf.io.read_file(filepath))
#         images.append(image)
#         labels.append(a_labels)
#     # resizing
#     images = [
#         tf.image.resize(
#             image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
#         )
#         for image in images
#     ]
#     fergus_labels = [1 if "fergus" in label else 0 for label in labels]
#     dataset = tf.data.Dataset.from_tensor_slices((images, fergus_labels))
#     return dataset


def transformed_name(key):
    return key + "_xf"


def _gzip_reader_fn(filenames):
    """
    gzip reader as pipeline stores intermediary data as gzipped tfrecords.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern,
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size=32,
):
    """
    Function to load from tfrecords in pipeline output.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key="labels_xf"),
        schema=tf_transform_output.transformed_metadata.schema,
    ).repeat()

    # return data_accessor.tf_dataset_factory(
    #     file_pattern,
    #     dataset_options.TensorFlowDatasetOptions(
    #         batch_size=batch_size, label_key=transformed_name(LABEL_KEY)
    #     ),
    #     tf_transform_output.transformed_metadata.schema,
    # ).repeat()

    # transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # dataset = tf.data.experimental.make_batched_features_dataset(
    #     file_pattern=file_pattern,
    #     batch_size=batch_size,
    #     features=transformed_feature_spec,
    #     reader=_gzip_reader_fn,
    #     label_key=transformed_name(LABEL_KEY),
    # )

    # return dataset


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
    outputs = tf.keras.layers.Dense(32, activation=tf.nn.softmax)(x)
    finetune_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return finetune_model


def run_fn(fn_args):
    """
    This may be only strictly neccassry function.
    """
    # training_dataset = _input_fn()
    # a wrapper around the output of tf.Transform.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    # training_dataset = input_fn()

    finetune_model = _build_keral_model()
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy()
    )
    history = finetune_model.fit(train_dataset, epochs=10, steps_per_epoch=100)
    finetune_model.save(
        fn_args.serving_model_dir, save_format="tf"
    )  # TODO: serving signature
