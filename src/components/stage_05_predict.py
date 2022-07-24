import logging
import os
import sys
import argparse

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

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)
    parser.add_argument('--test-image', type=str, help='path of dataset', required=True)

    return parser

class Prediction:
    def __init__(
        self,
        folder_path_config=PathConfig,
        param_config=ParameterConfig,
        test_image=None        
    ):
        logger.info(">>>>>Prediction using detectron2 started<<<<<<")
        self.folder_path_config = folder_path_config
        self.param_config = param_config
        self.predictor = Predictor(
            self.folder_path_config, self.param_config          
        )
        self.test_image = test_image

    def predict(self, test_image: 'str') -> None:
        """
        This function performs object detection(prediction) on a single image.
        Args:
            param test_image: path of the test image to be predicted
        Return: None
        """
        try:
            self.predictor.configure_predict_dataset()
            self.predictor.predict_image(test_image)
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    project_config = Configuration()
    predictor = Prediction(
        project_config.get_paths_config(),
        project_config.get_param_config(),
    )
    predictor.predict(args.test_image)
