import logging
import os
import sys
from src.config import Configuration
from src.exception import CustomException
from src.utils import Predictor
from src.entity import PathConfig, UrlNameConfig, ParameterConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Prediction:
    def __init__(
        self,
        folder_path_config=PathConfig,
        param_config=ParameterConfig,
    ):
        logger.info(">>>>>Prediction using detectron2 started<<<<<<")
        self.folder_path_config = folder_path_config
        self.param_config = param_config
        self.predictor = Predictor(
            self.folder_path_config, self.param_config
        )

    def predict(self) -> None:
        """
        This function performs object detection(prediction) on a single image.
        Return: None
        """
        try:
            self.predictor.configure_predict_dataset()
            self.predictor.predict_image()
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    predictor = Prediction(
        project_config.get_paths_config(),
        project_config.get_param_config(),
    )
    predictor.predict()
