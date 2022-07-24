import os
import logging
import sys

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


class Trainer:
    def __init__(
        self,
        folder_path_config=PathConfig,
        folder_url_name_config=UrlNameConfig,
        param_config=ParameterConfig,
    ):
        logger.info(">>>>>Register Dataset<<<<<<")
        self.path_config = folder_path_config
        self.url_config = folder_url_name_config
        self.param_config = param_config
        self.cfg = get_cfg()
        self.dataset_name = dict()

    def register_dataset(self, image_type=["train", "validation"]) -> None:
        """
        This function Registers the dataset.
        Return
        """
        try:
            logger.info(">>>>>Dataset registration started<<<<<<")
            for type in image_type:
                image_path = os.path.join(self.path_config.image_path, type, "image")
                coco_json_name = os.path.join(image_path, type + ".json")
                self.dataset_name[type] = "deepfashion_" + type
                register_coco_instances(
                    "deepfashion_" + type,
                    {},
                    coco_json_name,
                    os.path.join(self.path_config.image_path, type, "image"),
                )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Dataset registration finished<<<<<<")

    def configure_train_dataset(self) -> None:
        """
        This function create config for training.
        Return
        """
        try:
            logger.info(">>>>>Create config for train<<<<<<")
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.merge_from_file(
                model_zoo.get_config_file(self.url_config.model_url)
            )
            self.cfg.OUTPUT_DIR = self.path_config.output_path
            self.cfg.DATASETS.TRAIN = (self.dataset_name["validation"],)  ## chnage to "train" if you have good PC configuration
            self.cfg.DATASETS.TEST = ()
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                self.url_config.model_url
            )  # Let training initialize from model zoo

            self.cfg.SOLVER.IMS_PER_BATCH = self.param_config.images_per_batch
            self.cfg.SOLVER.BASE_LR = self.param_config.base_lr
            self.cfg.SOLVER.WARMUP_ITERS = self.param_config.warmup_iters
            self.cfg.SOLVER.MAX_ITER = self.param_config.max_iter
            self.cfg.SOLVER.STEPS = []
            self.cfg.SOLVER.GAMMA = self.param_config.gamma
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
                self.param_config.batch_size_per_image
            )
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.param_config.num_classes

            self.cfg.TEST.EVAL_PERIOD = self.param_config.eval_period
            logger.info(">>>>>Create config for train is finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Dataset registration finished<<<<<<")

    def train_model(self) -> None:
        """
        This function performs custom training.
        Return: None
        """
        try:
            logger.info(f">>>>>Load train config<<<<<<")
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            logger.info(f">>>>>Training started<<<<<<")
            trainer.train()
            logger.info(f">>>>>Training finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    convert = Trainer()
