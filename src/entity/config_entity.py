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
    ],
)
