import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    """
    A class to handle prediction pipeline for image classification.
    """
    def __init__(self, filename: str):
        """
        Initialize the PredictionPipeline with the image filename.
        
        Parameters:
            filename (str): The path to the image file.
        """
        self.filename = filename

    def predict(self) -> list:
        """
        Predict the class of the image.
        
        Returns:
            list: A list containing the prediction result.
        """
        try:
            # Load the model
            # model = load_model(os.path.join("artifacts", "training", "model.h5"))
            model = load_model(os.path.join("model", "model.h5"))

            # Load and preprocess the image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            # Perform prediction
            result = np.argmax(model.predict(test_image), axis=1)

            # Interpret result
            prediction = 'Normal' if result[0] == 1 else 'Adenocarcinoma Cancer'
            return [{"image": prediction}]

        except Exception as e:
            print(f"An error occurred: {e}")
            return [{"image": "Error"}]
