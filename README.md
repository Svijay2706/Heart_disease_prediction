
# ❤️ Heart Disease Prediction using Machine Learning

<p align="center">
  <b>An End-to-End Machine Learning Web Application for Heart Disease Risk Prediction</b>
</p>

<p align="center">
  Built with Python • Scikit-learn • Flask • Pandas • NumPy • HTML/CSS
</p>

---

## 📌 Project Overview

**Heart Disease Prediction** is an end-to-end Machine Learning project designed to predict whether a patient is likely to have heart disease based on clinical and demographic parameters.

The project implements a complete machine learning workflow:

**Data Loading → Data Validation → Preprocessing → Feature Scaling → Model Comparison → Cross-Validation → ROC-AUC Evaluation → Best Model Selection → Model Training → Prediction → Flask Web Application**

The project also demonstrates software engineering concepts such as **Object-Oriented Programming (OOP), exception handling, logging, modular code structure, model serialization, and web deployment readiness.**

> **Disclaimer:** This project is developed for educational and demonstration purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 👨‍💻 Developer

**Sakhamuri Vijay Bhargav**

**Role:** Data Scientist Intern

**Organization:** ViharaTech

**Program:** Data Science with Agentic AI

---

## 🎯 Project Objectives

* Build a machine learning model for heart disease classification.
* Perform proper data preprocessing and validation.
* Compare multiple machine learning algorithms.
* Evaluate models using **ROC-AUC** and accuracy.
* Apply **Stratified K-Fold Cross-Validation**.
* Select the best-performing model based on cross-validated ROC-AUC.
* Save the trained model and scaler for future predictions.
* Build a Flask-based web interface.
* Implement reusable and modular Python classes.
* Demonstrate logging and exception handling.
* Prepare the application for cloud deployment.

---

## 📊 Dataset

The project uses the **Heart Disease Dataset** containing **303 patient records** and **14 columns**.

### Dataset Source

Kaggle:

https://www.kaggle.com/datasets/yasserh/heart-disease-dataset

### Features

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `age`      | Age of the patient                |
| `sex`      | Gender                            |
| `cp`       | Chest pain type                   |
| `trestbps` | Resting blood pressure            |
| `chol`     | Serum cholesterol                 |
| `fbs`      | Fasting blood sugar               |
| `restecg`  | Resting ECG results               |
| `thalach`  | Maximum heart rate achieved       |
| `exang`    | Exercise-induced angina           |
| `oldpeak`  | ST depression induced by exercise |
| `slope`    | Slope of peak exercise ST segment |
| `ca`       | Number of major vessels           |
| `thal`     | Thalassemia                       |
| `target`   | Heart disease indicator           |

### Target Variable

```text
0 → No Heart Disease
1 → Heart Disease
```

---

# 🧠 Machine Learning Approach

The project compares multiple classification algorithms instead of relying on a single model.

### Models Evaluated

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

---

## 🔬 Model Evaluation

The models are evaluated using:

* Accuracy
* ROC-AUC
* 5-Fold Stratified Cross-Validation
* Classification Report
* Confusion Matrix
* ROC Curve

### Why ROC-AUC?

ROC-AUC measures how effectively a binary classification model separates the two classes across different classification thresholds.

A higher AUC indicates better class discrimination on the evaluated data.

---

## 🔄 Cross-Validation

The project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

and:

```python
cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)
```

### Why StratifiedKFold?

Because this is a binary classification problem, stratification helps maintain a similar proportion of the two target classes across the folds.

### Why Cross-Validation?

Instead of depending on only one train/test split, the model is evaluated across multiple folds to obtain a more reliable estimate of performance.

The model selection process uses the **mean cross-validated ROC-AUC**.

---

# 🌲 Model Selection

The project automatically compares the five algorithms and selects the model with the highest **mean 5-fold ROC-AUC** among the tested models.

For the current dataset and evaluation configuration, **Random Forest** was selected as the final model based on the cross-validation results.

> This selection is specific to the dataset and evaluation setup used in this project and should not be interpreted as a universal statement that Random Forest is the best algorithm for heart disease prediction.

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │    heart.csv        │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │   Data Preprocessing    │
                 │   Data Validation       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Train / Test Split   │
                 │    StandardScaler       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Model Comparison     │
                 │                         │
                 │ Logistic Regression     │
                 │ Random Forest           │
                 │ Gradient Boosting       │
                 │ SVM                     │
                 │ KNN                     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Stratified 5-Fold CV    │
                 │ ROC-AUC Evaluation      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Best Model Selected  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Model Evaluation      │
                 │ Accuracy / ROC-AUC      │
                 │ Confusion Matrix        │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       Joblib            │
                 │ heart_model.pkl         │
                 │ scaler.pkl              │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Flask Web App      │
                 │     Patient Prediction  │
                 └─────────────────────────┘
```

---

# 📁 Project Structure

```text
Heart-Disease-Prediction-ML/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── heart_model.pkl
│   └── scaler.pkl
│
├── results/
│   └── roc_curve_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
│
├── static/
│   ├── images/
│   │   ├── vijay.jpg
│   │   └── vihara-logo.png
│   └── style.css
│
├── templates/
│   └── index.html
│
├── logs/
│   └── application.log
│
├── app.py
├── main.py
├── test_prediction.py
├── Procfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧩 OOP Implementation

The project is structured using Python classes to improve modularity and maintainability.

### `DataPreprocessor`

Responsible for:

* Loading the dataset
* Validating data
* Checking missing values
* Inspecting data types
* Logging preprocessing operations

### `HeartDiseaseModel`

Responsible for:

* Preparing training/testing data
* Creating ML models
* Comparing algorithms
* Cross-validation
* ROC-AUC evaluation
* Selecting the best model
* Training the final model
* Saving the model

### `PredictionService`

Responsible for:

* Receiving patient information
* Creating a DataFrame
* Applying the saved scaler
* Generating predictions
* Returning prediction probabilities

### `Logger`

Responsible for:

* Application logs
* Information messages
* Warning messages
* Error messages

---

# 💾 Model Serialization

The project uses **Joblib** to save trained machine learning components.

```python
joblib.dump(self.model, "models/heart_model.pkl")
joblib.dump(self.scaler, "models/scaler.pkl")
```

This allows the Flask application to load the already-trained model without retraining every time the application starts.

The scaler is also saved because new patient data must undergo the **same feature transformation** used during model training.

---

# 📝 Logging

Application activity is recorded in:

```text
logs/application.log
```

Examples of logged operations include:

```text
Dataset loading
Dataset validation
Model training
Model comparison
Model evaluation
Prediction requests
Prediction errors
Model saving
```

---

# ⚠️ Exception Handling

The project uses Python exception handling to improve reliability.

Examples include:

```python
try:
    ...
except FileNotFoundError:
    ...
except ValueError:
    ...
except Exception as e:
    ...
```

This helps handle issues such as:

* Missing dataset
* Invalid user input
* Model loading errors
* Prediction errors
* File-related errors
* Unexpected runtime errors

---

# 🌐 Flask Web Application

The trained model is integrated into a Flask web application.

### Application Flow

```text
User
  ↓
Web Interface
  ↓
Flask
  ↓
PredictionService
  ↓
Scaler
  ↓
Random Forest Model
  ↓
Prediction Probability
  ↓
Web Interface
```

The application accepts patient parameters and displays:

* Prediction result
* Heart disease probability
* No heart disease probability

---

# 🖥️ Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction-ML.git
```

```bash
cd Heart-Disease-Prediction-ML
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚂 Train the Machine Learning Pipeline

Run:

```bash
python main.py
```

The pipeline will:

```text
Load Dataset
      ↓
Validate Dataset
      ↓
Split Data
      ↓
Scale Features
      ↓
Compare Models
      ↓
5-Fold ROC-AUC
      ↓
Select Best Model
      ↓
Evaluate Model
      ↓
Save Model
```

The trained files will be created in:

```text
models/
├── heart_model.pkl
└── scaler.pkl
```

The ROC curve comparison will be saved as:

```text
results/roc_curve_comparison.png
```

---

# 🧪 Test Prediction

Run:

```bash
python test_prediction.py
```

The script loads:

```text
heart_model.pkl
scaler.pkl
```

and generates a sample prediction with class probabilities.

---

# 🚀 Run the Flask Application

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

For a production-style local server using Waitress:

```bash
python -m waitress --listen=127.0.0.1:8000 app:app
```

---

# ☁️ Deployment

The project is structured for deployment on platforms such as **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

The project also includes:

```text
Procfile
```

containing:

```text
web: gunicorn app:app
```
# 🌐 Project Deployment

### Live Demo

🔗 **[https://heart-disease-prediction-y5k5.onrender.com]**

### GitHub Repository

🔗 **[https://github.com/Svijay2706/Heart_disease_prediction]**
---

# 🛠️ Technologies Used

| Category                | Technologies  |
| ----------------------- | ------------- |
| Programming Language    | Python        |
| Data Processing         | Pandas, NumPy |
| Machine Learning        | Scikit-learn  |
| Visualization           | Matplotlib    |
| Model Persistence       | Joblib        |
| Web Framework           | Flask         |
| Frontend                | HTML, CSS     |
| Version Control         | Git, GitHub   |
| Deployment              | Render        |
| Development Environment | VS Code       |

---

# 📈 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Machine Learning
* Binary Classification
* Data Preprocessing
* Feature Scaling
* Train-Test Split
* Logistic Regression
* Random Forest
* Gradient Boosting
* SVM
* KNN
* ROC Curve
* ROC-AUC
* Stratified K-Fold Cross-Validation
* Model Selection
* Model Serialization
* Object-Oriented Programming
* Exception Handling
* Logging
* Flask
* REST-style prediction endpoint
* Git & GitHub
* Cloud Deployment

---

# 🔮 Future Improvements

Possible future enhancements include:

* Hyperparameter optimization
* Feature importance visualization
* SHAP-based model explainability
* Interactive prediction dashboard
* Improved input validation
* Database integration
* User authentication
* Prediction history
* Docker containerization
* CI/CD pipeline
* Cloud monitoring
* Model performance monitoring

---

# 📌 Important Medical Disclaimer

This application is an **educational machine learning project** and should not be used as a medical diagnostic system.

The predictions generated by this application are based on the training dataset and machine learning model and should not be treated as professional medical advice.

For medical concerns, users should consult a qualified healthcare professional.

---

# 👨‍💻 Author

### Sakhamuri Vijay Bhargav

**Data Scientist Intern | Python | Machine Learning | Data Science | AI**

This project was developed as part of my **Data Science with Agentic AI training/internship at ViharaTech**.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and demonstration purposes.
