import logging
import os
import sys
import argparse

from src.config import Configuration
from src.exception import CustomException
from src.utils import Trainer
from src.entity import PathConfig, UrlNameConfig, ParameterConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Custom_Train:
    def __init__(
        self,
        folder_path_config=PathConfig,
        folder_url_name_config=UrlNameConfig,
        param_config=ParameterConfig,
    ):
        logger.info(">>>>>Detectron2 Custom Model Training preparation started<<<<<<")
        self.folder_path_config = folder_path_config
        self.folder_url_name_config = folder_url_name_config
        self.param_config = param_config
        self.trainer = Trainer(
            self.folder_path_config, self.folder_url_name_config, self.param_config
        )

    def train_model(self) -> None:
        """
        This function performs dataset register, configur train dataset and train model.
        Return: None
        """
        try:
            self.trainer.register_dataset()
            self.trainer.configure_train_dataset()
            self.trainer.train_model()
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    custom_train = Custom_Train(
        project_config.get_paths_config(),
        project_config.get_url_name_config(),
        project_config.get_param_config(),
    )
    custom_train.train_model()
