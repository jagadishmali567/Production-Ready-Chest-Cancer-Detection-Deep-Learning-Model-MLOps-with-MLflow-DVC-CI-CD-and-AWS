import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training class with the given configuration.

        Args:
            config (TrainingConfig): Configuration parameters for training.
        """
        self.config = config  # Store the training configuration

    def get_base_model(self):
        """
        Loads the base model from the specified path.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Creates training and validation data generators with or without augmentation
        based on the configuration parameters.
        """
        # Common data generator arguments
        datagenerator_kwargs = dict(
            rescale=1./255,  # Rescale pixel values
            validation_split=0.20  # Split 20% of data for validation
        )

        # Arguments for data flow
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Target image size
            batch_size=self.config.params_batch_size,  # Batch size
            interpolation="bilinear"  # Interpolation method
        )

        # Create validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Flow validation data from directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Conditional data augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Rotate images
                horizontal_flip=True,  # Flip images horizontally
                width_shift_range=0.2,  # Shift images horizontally
                height_shift_range=0.2,  # Shift images vertically
                shear_range=0.2,  # Shear images
                zoom_range=0.2,  # Zoom images
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator  # Use validation generator if no augmentation

        # Flow training data from directory
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained model to the specified path.

        Args:
            path (Path): Path where the model will be saved.
            model (tf.keras.Model): Trained TensorFlow model.
        """
        model.save(path)

    def train(self):
        """
        Trains the model using the training and validation generators.

        Steps are calculated based on the number of samples and batch size.
        The trained model is saved to the specified path.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
