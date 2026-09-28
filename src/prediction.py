import pandas as pd

from src.logger import Logger


class PredictionService:

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.logger = Logger()

        self.feature_names = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]

    def predict(self, patient_data):

        try:
            self.logger.info("Prediction request received")

            # Convert patient input into DataFrame
            patient_df = pd.DataFrame(
                [patient_data],
                columns=self.feature_names
            )

            # Scale the input
            scaled_data = self.scaler.transform(patient_df)

            # Make prediction
            prediction = self.model.predict(scaled_data)[0]

            # Get probabilities
            probability = self.model.predict_proba(scaled_data)[0]

            if prediction == 1:
                result = "Heart Disease Detected"
            else:
                result = "No Heart Disease Detected"

            self.logger.info(
                f"Prediction completed: {result}"
            )

            return result, probability

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise