import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Prepares the configuration for the base model.

        Reads the necessary settings from the configuration and parameters,
        creates the required directories, and returns a PrepareBaseModelConfig object.

        Returns:
            PrepareBaseModelConfig: Configuration object for preparing the base model.
        """
        config = self.config.prepare_base_model
        
        # Create the root directory specified in the base model configuration
        create_directories([config.root_dir])

        # Prepare the base model configuration using the settings from the configuration and parameters
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        """
        Creates a TrainingConfig object with the training parameters and file paths.

        Returns:
            TrainingConfig: An instance of the TrainingConfig class containing training configuration details.
        """
        training = self.config.training  # Retrieve training configuration
        prepare_base_model = self.config.prepare_base_model  # Retrieve base model preparation configuration
        params = self.params  # Retrieve training parameters

        # Construct the path to the training data
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")

        # Create the directory for the training root if it doesn't exist
        create_directories([Path(training.root_dir)])  

        # Create an instance of TrainingConfig with the retrieved configurations and parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves the evaluation configuration.

        Returns:
            EvaluationConfig: An instance of EvaluationConfig with the specified settings.
        """
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",  # Path to the trained model
            training_data="artifacts/data_ingestion/Chest-CT-Scan-data",  # Path to the training data
            mlflow_uri="https://dagshub.com/jagadishmali567/Production-Ready-Chest-Cancer-Detection-Deep-Learning-Model-MLOps-with-MLflow-DVC-CI-CD-and-AWS.mlflow",  # MLflow tracking URI
            all_params=self.params,  # Parameters from the parameters YAML file
            params_image_size=self.params.IMAGE_SIZE,  # Image size parameter
            params_batch_size=self.params.BATCH_SIZE  # Batch size parameter
        )
        return eval_config