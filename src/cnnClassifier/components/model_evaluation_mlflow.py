import tensorflow as tf
from pathlib import Path
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    """
    Class to handle the evaluation process of a trained machine learning model.

    Attributes:
        config (EvaluationConfig): Configuration settings for model evaluation.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the Evaluation with the given configuration.

        Args:
            config (EvaluationConfig): Configuration settings for model evaluation.
        """
        self.config = config

    def _valid_generator(self):
        """
        Creates a validation data generator using Keras' ImageDataGenerator.

        This method sets up the data augmentation and preprocessing steps and
        creates a flow from the directory specified in the configuration.
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a trained Keras model from the specified file path.

        Args:
            path (Path): Path to the trained model file.

        Returns:
            tf.keras.Model: Loaded Keras model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Evaluates the trained model using the validation data generator.

        This method loads the model, sets up the validation data generator,
        performs evaluation, and saves the evaluation scores.
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Saves the evaluation scores to a JSON file.

        This method creates a dictionary of scores and saves it as a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Logs the evaluation parameters and metrics into MLflow.

        This method logs the parameters, evaluation scores, and registers the
        model in MLflow if applicable.
        """
        # Initialize DagsHub for MLflow tracking
        dagshub.init(repo_owner='jagadishmali567', 
                     repo_name='Production-Ready-Chest-Cancer-Detection-Deep-Learning-Model-MLOps-with-MLflow-DVC-CI-CD-and-AWS', 
                     mlflow=True)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # Register the model
            mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
