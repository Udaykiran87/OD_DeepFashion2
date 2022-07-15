import logging
import os
import sys
import wget
import subprocess
from src.config import Configuration
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Download_Dataset:
    def __init__(
        self, folder_path_config=PathConfig, folder_url_name_config=UrlNameConfig
    ):
        logger.info(">>>>>Download of DeepFashion2-Dataset<<<<<<")
        self.path_config = folder_path_config
        self.url_name_config = folder_url_name_config

    def download_data(self) -> None:
        """
        This function downloads DeepFashion2-Dataset.
        Returns: None
        """
        logger.info(">>>>>Download of DeepFashion2-Dataset started<<<<<<")
        is_data_absent = False
        try:
            if os.name == "posix":
                logger.info(f">>>>>>Posix System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "train.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "test.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "validation.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "json_for_validation.zip",
                    )
                ):
                    is_data_absent = True

                if is_data_absent:
                    logger.info(">>>>>Download of DeepFashion2-Dataset  started<<<<<<")
                    cmd = f"gdown {self.url_name_config.dataset_train_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"mv train.zip {os.path.join(self.path_config.image_path)}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** train.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.dataset_test_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"mv test.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** test.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.dataset_val_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"mv validation.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** validation.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.json_for_val_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"mv json_for_validation.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** json_for_validation.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(">>>>>Download of DeepFashion2-Dataset  finished<<<<<<")
                else:
                    logger.info(
                        f">>>>>Download not required as dataset is already present at {self.path_config.image_path}<<<<<<"
                    )

            if os.name == "nt":
                logger.info(f">>>>>>Nt System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "train.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "test.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "validation.zip",
                    )
                ):
                    is_data_absent = True

                if not os.path.exists(
                    os.path.join(
                        self.path_config.image_path,
                        "json_for_validation.zip",
                    )
                ):
                    is_data_absent = True

                if is_data_absent:
                    logger.info(">>>>>Download of DeepFashion2-Dataset  started<<<<<<")
                    cmd = f"gdown {self.url_name_config.dataset_train_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"move train.zip {os.path.join(self.path_config.image_path)}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** train.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.dataset_test_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"move test.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** test.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.dataset_val_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"move validation.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** validation.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")

                    cmd = f"gdown {self.url_name_config.json_for_val_url}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"move json_for_validation.zip {self.path_config.image_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.image_path} && unzip -P 2019Deepfashion2** json_for_validation.zip"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(">>>>>Download of DeepFashion2-Dataset  finished<<<<<<")
                else:
                    logger.info(
                        f">>>>>Download not required as dataset is already present at {self.path_config.image_path}<<<<<<"
                    )

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Download of DeepFashion2-Dataset finished<<<<<<")


if __name__ == "__main__":
    project_config = Configuration()
    download = Download_Dataset(
        project_config.get_paths_config(), project_config.get_url_name_config()
    )
    download.download_data()
