from flask import Flask, render_template, request

import joblib

from src.prediction import PredictionService


app = Flask(__name__)


# Load trained model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

predictor = PredictionService(model, scaler)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        patient_data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        result, probability = predictor.predict(patient_data)

        heart_probability = probability[1] * 100
        no_heart_probability = probability[0] * 100

        return render_template(
            "index.html",
            prediction=result,
            heart_probability=f"{heart_probability:.2f}",
            no_heart_probability=f"{no_heart_probability:.2f}"
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"An error occurred: {e}"
        )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
        use_reloader=False
    )