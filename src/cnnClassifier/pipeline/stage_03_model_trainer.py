from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        """
        Initializes the ModelTrainingPipeline class.
        """
        pass

    def main(self):
        """
        Main method to execute the model training pipeline.
        """
        config = ConfigurationManager()  # Initialize configuration manager
        training_config = config.get_training_config()  # Get training configuration
        training = Training(config=training_config)  # Initialize training with configuration
        
        training.get_base_model()  # Load the base model
        training.train_valid_generator()  # Create training and validation data generators
        training.train()  # Train the model

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        obj = ModelTrainingPipeline()  # Create an instance of the pipeline
        obj.main()  # Execute the main method
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Re-raise the exception
