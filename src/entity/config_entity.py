from collections import namedtuple

PathConfig = namedtuple(
    "PathConfig",
    [
        "workspace_path",
        "image_path",
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
    ],
)
