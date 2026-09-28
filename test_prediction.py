import joblib

from src.prediction import PredictionService


def main():

    try:
        # Load trained model and scaler
        model = joblib.load("models/heart_model.pkl")
        scaler = joblib.load("models/scaler.pkl")

        # Create prediction service
        predictor = PredictionService(model, scaler)

        # Sample patient data
        patient_data = [
    37, 1, 2, 130, 250, 0, 1,
    187, 0, 3.5, 0, 0, 2
]

        result, probability = predictor.predict(patient_data)

        print("\nPrediction Result")
        print("-----------------")
        print(result)

        print("\nPrediction Probability")
        print("----------------------")
        print(f"No Heart Disease: {probability[0] * 100:.2f}%")
        print(f"Heart Disease: {probability[1] * 100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()