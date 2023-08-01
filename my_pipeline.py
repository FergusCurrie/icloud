import os
from typing import Optional, Text, List
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from tfx.components import (
    ImportExampleGen,
    StatisticsGen,
    SchemaGen,
    Transform,
    Trainer,
)
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.proto import trainer_pb2
import tensorflow_data_validation as tfdv

from absl import logging

logging.set_verbosity(logging.INFO)

PIPELINE_NAME = "my_pipeline"
PIPELINE_ROOT = os.path.join(".", "my_pipeline_output")
METADATA_PATH = os.path.join(".", "tfx_metadata", PIPELINE_NAME, "metadata.db")
ENABLE_CACHE = True


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: bool,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
):
    example_gen = ImportExampleGen(input_base="tfrecord")

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath("tft_module.py"),
    )

    trainer = Trainer(
        module_file="tfx_trainer.py",
        examples=transform.outputs[
            "transformed_examples"
        ],  # TODO: is this a set string
        transform_graph=transform.outputs["transform_graph"],
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
    )

    components = [example_gen, statistics_gen, schema_gen, transform, trainer]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline():
    my_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    tfx.orchestration.LocalDagRunner().run(my_pipeline)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()
