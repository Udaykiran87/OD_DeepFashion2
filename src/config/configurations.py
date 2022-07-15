import os
import logging
import sys

from src.utils import read_yaml
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


class Configuration:
    def __init__(self):
        try:
            logger.info(">>>>>>Reading Config file<<<<<<")
            self.config = read_yaml(path_to_yaml=CONFIG_FILE_NAME)
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def get_paths_config(self) -> PathConfig:
        """This function reads all directory configurations.
        Returns: PathConfig object.
        """
        try:
            workspace_dir = self.config[PATH_KEY][WORKSPACE_NAME]
            image_dir = self.config[PATH_KEY][IMAGE_PATH]

            artifcats_path = self.config[ARTIFACTS_KEY][ARTIFACTS_DIR]
            workspace_path = os.path.join(
                artifcats_path, workspace_dir
            )

            image_path = os.path.join(workspace_path, image_dir)

            folder_path_config = PathConfig(
                workspace_path=workspace_path,
                image_path=image_path,
            )
            return folder_path_config
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def get_url_name_config(self) -> UrlNameConfig:
        """This function reads all url, name configurations.
        Returns: UrlNameConfig object.
        """
        try:
            detectron2_url = self.config[ARTIFACTS_KEY][DETECTRON2_URL]
            detectron2_url = self.config[ARTIFACTS_KEY][DETECTRON2_URL]

            url_name_config = UrlNameConfig(
                detectron2_url=detectron2_url,
            )
            return url_name_config
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = Configuration()
