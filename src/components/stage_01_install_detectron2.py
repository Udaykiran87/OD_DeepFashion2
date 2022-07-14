import logging
import sys
import os
import subprocess
from src.config import Configuration
from src.constants import *
from src.exception import CustomException
from src.entity import UrlNameConfig

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

class Detectron2_Install:
    def __init__(self, url_name_config=UrlNameConfig):
        logger.info(">>>>>Detectron2 Installation started<<<<<<")
        self.url_name_config = url_name_config

    def install(self) -> None:
        """
        This function installs detectron2.
        Returns: None
        """
        logger.info(">>>>>Installation of detectron2 started<<<<<<")
        try:
            cmd = f"git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 && git reset --hard 453d795eb152f7210c842674810f24769d4dd8cb"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            cmd = f"cp scripts/nms_rotated_cuda.cu detectron2/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            cmd = f"cd detectron2 && pip install -e ."
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Installation of detectron2 finished<<<<<<")

if __name__ == "__main__":
    project_config = Configuration()
    detectron2 = Detectron2_Install(project_config.get_url_name_config())
    detectron2.install()