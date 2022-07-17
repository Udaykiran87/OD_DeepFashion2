from PIL import Image
import numpy as np
import json
import os
import logging
import sys

from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

lst_name = [
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeved_dress",
    "long_sleeved_dress",
    "vest_dress",
    "sling_dress",
]

num_images = {
    "train": 191961,
    "validation": 32153,
}


class Converter:
    def __init__(self, folder_path_config=PathConfig):
        logger.info(">>>>>Convert dataset into coco format<<<<<<")
        self.path_config = folder_path_config

    def to_coco(self, image_type=["train", "validation"]) -> None:
        """
        This function converts train, validation dataset into coco format.
        Return
        """
        try:
            logger.info(">>>>>Coco coversion started<<<<<<")
            for type in image_type:
                json_path = os.path.join(self.path_config.image_path, type, "annos")
                image_path = os.path.join(self.path_config.image_path, type, "image")
                img_count = num_images[type]
                sub_index = 0  # the index of ground truth instance
                dataset = {
                    "info": {},
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": [],
                }

                for idx, e in enumerate(lst_name):
                    dataset["categories"].append(
                        {
                            "id": idx + 1,
                            "name": e,
                            "supercategory": "clothes",
                            "keypoints": ["%i" % (i) for i in range(1, 295)],
                            "skeleton": [],
                        }
                    )

                for num in range(1, img_count + 1):
                    json_name = os.path.join(json_path, str(num).zfill(6) + ".json")
                    image_name = os.path.join(image_path, str(num).zfill(6) + ".jpg")
                    if num >= 0:
                        imag = Image.open(image_name)
                        width, height = imag.size
                        with open(json_name, "r") as f:
                            temp = json.loads(f.read())
                            pair_id = temp["pair_id"]

                            dataset["images"].append(
                                {
                                    "coco_url": "",
                                    "date_captured": "",
                                    "file_name": str(num).zfill(6) + ".jpg",
                                    "flickr_url": "",
                                    "id": num,
                                    "license": 0,
                                    "width": width,
                                    "height": height,
                                }
                            )
                            for i in temp:
                                if i == "source" or i == "pair_id":
                                    continue
                                else:
                                    points = np.zeros(294 * 3)
                                    sub_index = sub_index + 1
                                    box = temp[i]["bounding_box"]
                                    w = box[2] - box[0]
                                    h = box[3] - box[1]
                                    x_1 = box[0]
                                    y_1 = box[1]
                                    bbox = [x_1, y_1, w, h]
                                    cat = temp[i]["category_id"]
                                    style = temp[i]["style"]
                                    seg = temp[i]["segmentation"]
                                    landmarks = temp[i]["landmarks"]

                                    points_x = landmarks[0::3]
                                    points_y = landmarks[1::3]
                                    points_v = landmarks[2::3]
                                    points_x = np.array(points_x)
                                    points_y = np.array(points_y)
                                    points_v = np.array(points_v)
                                    case = [
                                        0,
                                        25,
                                        58,
                                        89,
                                        128,
                                        143,
                                        158,
                                        168,
                                        182,
                                        190,
                                        219,
                                        256,
                                        275,
                                        294,
                                    ]
                                    idx_i, idx_j = case[cat - 1], case[cat]

                                    for n in range(idx_i, idx_j):
                                        points[3 * n] = points_x[n - idx_i]
                                        points[3 * n + 1] = points_y[n - idx_i]
                                        points[3 * n + 2] = points_v[n - idx_i]

                                    num_points = len(np.where(points_v > 0)[0])

                                    dataset["annotations"].append(
                                        {
                                            "area": w * h,
                                            "bbox": bbox,
                                            "category_id": cat,
                                            "id": sub_index,
                                            "pair_id": pair_id,
                                            "image_id": num,
                                            "iscrowd": 0,
                                            "style": style,
                                            "num_keypoints": num_points,
                                            "keypoints": points.tolist(),
                                            "segmentation": seg,
                                        }
                                    )

                coco_json_name = os.path.join(image_path, type + ".json")
                with open(coco_json_name, "w") as f:
                    json.dump(dataset, f)
                logger.info(
                    f">>>>>Converted coco file for {type} is saved at {coco_json_name}.<<<<<<"
                )

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Download of DeepFashion2-Dataset finished<<<<<<")


if __name__ == "__main__":
    convert = Converter()
