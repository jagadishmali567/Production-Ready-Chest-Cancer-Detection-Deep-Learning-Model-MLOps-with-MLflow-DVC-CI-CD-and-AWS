from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    """
    A pipeline class to prepare the base model for training.

    Methods:
        main(): Main method to execute the pipeline stages.
    """
    
    def __init__(self):
        pass

    def main(self):
        """
        Main method to prepare and update the base model using the configuration.
        """
        # Initialize ConfigurationManager and get the prepare base model configuration
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        
        # Initialize PrepareBaseModel with the configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Load and save the base model
        prepare_base_model.get_base_model()
        
        # Update and save the base model
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the pipeline and run the main method
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception and raise it
        logger.exception(e)
        raise e
