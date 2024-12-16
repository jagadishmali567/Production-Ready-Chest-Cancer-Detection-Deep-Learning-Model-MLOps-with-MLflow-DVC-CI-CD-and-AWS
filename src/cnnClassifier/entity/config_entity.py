from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for preparing a base model.
    Attributes:
        root_dir (Path): Root directory for the project.
        base_model_path (Path): Path to the base model.
        updated_base_model_path (Path): Path to the updated base model.
        params_image_size (list): List containing image dimensions.
        params_learning_rate (float): Learning rate for the model.
        params_include_top (bool): Whether to include the top layers of the model.
        params_weights (str): Weights to be used for the model.
        params_classes (int): Number of output classes for the model.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    
    
@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration class for training parameters and file paths.

    Attributes:
        root_dir (Path): Directory where the project is located.
        trained_model_path (Path): Path to the trained model.
        updated_base_model_path (Path): Path to the updated base model.
        training_data (Path): Path to the training data.
        params_epochs (int): Number of training epochs.
        params_batch_size (int): Size of each training batch.
        params_is_augmentation (bool): Indicates if data augmentation is applied.
        params_image_size (list): Size of the images used for training.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Data class to hold configuration settings for model evaluation.

    Attributes:
        path_of_model (Path): Path to the trained model.
        training_data (Path): Path to the training data used for the model.
        all_params (dict): Dictionary containing all parameters used for model training and evaluation.
        mlflow_uri (str): URI for MLflow tracking server to log experiments and results.
        params_image_size (list): List containing the dimensions of the input images.
        params_batch_size (int): Size of the batches for evaluation.
    """
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int