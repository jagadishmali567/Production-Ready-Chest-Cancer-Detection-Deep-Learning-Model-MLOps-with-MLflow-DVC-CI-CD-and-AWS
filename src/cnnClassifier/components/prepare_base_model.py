import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    A class to prepare and update the base model using the given configuration.

    Attributes:
        config (PrepareBaseModelConfig): Configuration for preparing the base model.
        model (tf.keras.Model): The base model.
        full_model (tf.keras.Model): The updated full model.

    Methods:
        get_base_model(): Initializes the base model.
        update_base_model(): Updates the base model by adding custom layers and compiling.
        save_model(path, model): Saves the given model to the specified path.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel with the given configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration for preparing the base model.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the base model using the VGG16 architecture with the specified parameters
        and saves it to the configured path.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Save the base model to the specified path
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model by adding custom layers, setting trainable layers, and compiling.

        Args:
            model (tf.keras.Model): The base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int or None): Number of layers to keep unfrozen from the end.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The fully prepared model with custom layers added.
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Updates the base model by preparing and compiling the full model with custom layers.
        The updated model is then saved to the configured path.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated full model to the specified path
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given model to the specified path.

        Args:
            path (Path): The path where the model should be saved.
            model (tf.keras.Model): The model to save.
        """
        model.save(path)
