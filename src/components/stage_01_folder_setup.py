import logging
import sys
import os
from src.config import Configuration
from src.utils import create_directories
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Folder_Setup:
    def __init__(self, Folder_path_config=PathConfig, Folder_url_name_config=UrlNameConfig):
        logger.info(">>>>>Folder setup started<<<<<<")
        self.path_config = Folder_path_config
        self.url_name_config = Folder_url_name_config

    def create_dirs(self) -> None:
        """
        This function creates necessary directories or paths.
        Returns: None
        """
        try:
            logger.info(f">>>>>Directories creation started<<<<<<")
            create_directories(list(self.path_config))
            logger.info(f">>>>>Directories creation finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    Folder_setup = Folder_Setup(
        project_config.get_paths_config(), project_config.get_url_name_config()
    )
    Folder_setup.create_dirs()
