import os
import logging
import sys

from src.utils import read_yaml
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig, ParameterConfig

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
            output_dir = self.config[PATH_KEY][OUTPUT_PATH]

            artifcats_path = self.config[ARTIFACTS_KEY][ARTIFACTS_DIR]
            workspace_path = os.path.join(artifcats_path, workspace_dir)

            image_path = os.path.join(workspace_path, image_dir)
            output_path = os.path.join(workspace_path, output_dir)

            folder_path_config = PathConfig(
                workspace_path=workspace_path,
                image_path=image_path,
                output_path=output_path,
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
            dataset_train_url = self.config[ARTIFACTS_KEY][TRAIN_URL]
            dataset_test_url = self.config[ARTIFACTS_KEY][TEST_URL]
            dataset_val_url = self.config[ARTIFACTS_KEY][VALIDATION_URL]
            json_for_val_url = self.config[ARTIFACTS_KEY][JSON_FOR_VAL_URL]
            model_url = self.config[ARTIFACTS_KEY][MODEL_URL]

            url_name_config = UrlNameConfig(
                detectron2_url=detectron2_url,
                dataset_train_url=dataset_train_url,
                dataset_test_url=dataset_test_url,
                dataset_val_url=dataset_val_url,
                json_for_val_url=json_for_val_url,
                model_url=model_url,
            )
            return url_name_config
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def get_param_config(self) -> ParameterConfig:
        """This function reads training and test parameters.
        Returns: ParameterConfig object.
        """
        try:
            images_per_batch = self.config[PARAM_KEY][IMS_PER_BATCH]
            base_lr = self.config[PARAM_KEY][BASE_LR]
            warmup_iters = self.config[PARAM_KEY][WARMUP_ITERS]
            max_iter = self.config[PARAM_KEY][MAX_ITER]
            steps = self.config[PARAM_KEY][STEPS]
            gamma = self.config[PARAM_KEY][GAMMA]
            batch_size_per_image = self.config[PARAM_KEY][BATCH_SIZE_PER_IMAGE]
            num_classes = self.config[PARAM_KEY][NUM_CLASSES]
            eval_period = self.config[PARAM_KEY][EVAL_PERIOD]

            param_config = ParameterConfig(
                images_per_batch=int(images_per_batch),
                base_lr=float(base_lr),
                warmup_iters=int(warmup_iters),
                max_iter=int(max_iter),
                steps=(int(steps[0]), int(steps[1])),
                gamma=float(gamma),
                batch_size_per_image=int(batch_size_per_image),
                num_classes=int(num_classes),
                eval_period=int(eval_period),
            )
            return param_config
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = Configuration()
