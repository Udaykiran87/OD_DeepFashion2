import os
import logging
import sys
import cv2

from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig, ParameterConfig

from detectron2.data.datasets import register_coco_instances

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
import numpy as np
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        folder_path_config=PathConfig,
        param_config=ParameterConfig,
    ):
        self.path_config = folder_path_config
        self.param_config = param_config
        self.cfg = get_cfg()


    def configure_predict_dataset(self) -> None:
        """
        This function create config for prediction.
        Return
        """
        try:
            logger.info(">>>>>Create config for prediction<<<<<<")
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.DATASETS.TRAIN = ("deepfashion_val", )  ## chnage to "train" if you have good PC configuration
            self.cfg.DATASETS.TEST = ("deepfashion_val", )
            self.cfg.MODEL.WEIGHTS = os.path.join(self.path_config.output_path, "model_final.pth")
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.param_config.thresh_score_test   # set the testing threshold for this model            
            logger.info(">>>>>Create config for train is finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Dataset registration finished<<<<<<")

    def predict_image(self, test_image) -> None:
        """
        This function performs prediction on an image.
        Return: None
        """
        try:
            logger.info(f">>>>>Load predict config<<<<<<")
            predictor = DefaultPredictor(self.cfg)
            logger.info(f">>>>>Prediction started<<<<<<")
            # test_image = os.path.join(self.path_config.image_path, "test", "image", "000003.jpg")
            im = cv2.imread(test_image)
            outputs = predictor(im)
            logger.info(f">>>>>prediction finished<<<<<<")
            logger.info(f">>>>>visualization of test image started<<<<<<")
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("predicted_image", v.get_image()[:, :, ::-1])
            cv2.waitKey(0) 
            logger.info(f">>>>>visualization of test image finished<<<<<<")            
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    convert = Trainer()
