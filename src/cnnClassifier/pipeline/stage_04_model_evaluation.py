from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

import dagshub
import mlflow
import mlflow.keras

# Define the stage name for logging purposes
STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    """
    Pipeline class to handle the evaluation stage of the machine learning model.
    """

    def __init__(self):
        """
        Initializes the EvaluationPipeline instance.
        """
        pass

    def main(self):
        """
        Main method to execute the evaluation pipeline.

        This method initializes the configuration manager, retrieves the evaluation
        configuration, performs the evaluation, saves the evaluation scores, and logs
        the results into MLflow via DagsHub.
        """
        config = ConfigurationManager()  # Initialize configuration manager
        eval_config = config.get_evaluation_config()  # Retrieve evaluation configuration
        evaluation = Evaluation(eval_config)  # Initialize evaluation with the config
        evaluation.evaluation()  # Perform the evaluation
        evaluation.save_score()  # Save the evaluation scores

        # Initialize DagsHub for MLflow tracking
        #dagshub.init(repo_owner='jagadishmali567', 
                     #repo_name='Production-Ready-Chest-Cancer-Detection-Deep-Learning-Model-MLOps-with-MLflow-DVC-CI-CD-and-AWS', 
                     #mlflow=True)

        # Log the evaluation results into MLflow via DagsHub
        #with mlflow.start_run():
            #mlflow.log_params(eval_config.all_params)
            #mlflow.log_metrics({"loss": evaluation.score[0], "accuracy": evaluation.score[1]})
            #mlflow.keras.log_model(evaluation.model, "model", registered_model_name="VGG16Model")

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()  # Create an instance of EvaluationPipeline
        obj.main()  # Execute the main method
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Raise the exception for further handling
