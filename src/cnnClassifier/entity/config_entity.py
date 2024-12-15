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
        