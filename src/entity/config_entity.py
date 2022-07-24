from collections import namedtuple
from inspect import Parameter

PathConfig = namedtuple(
    "PathConfig",
    [
        "workspace_path",
        "image_path",
        "output_path",
    ],
)

UrlNameConfig = namedtuple(
    "UrlNameConfig",
    [
        "detectron2_url",
        "dataset_train_url",
        "dataset_test_url",
        "dataset_val_url",
        "json_for_val_url",
        "model_url",
    ],
)

ParameterConfig = namedtuple(
    "ParameterConfig",
    [
        "images_per_batch",
        "base_lr",
        "warmup_iters",
        "max_iter",
        "steps",
        "gamma",
        "batch_size_per_image",
        "num_classes",
        "eval_period",
        "thresh_score_test",
    ],
)
