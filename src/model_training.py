
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from src.logger import Logger


class HeartDiseaseModel:

    def __init__(self):

        # Logger
        self.logger = Logger()

        # Feature scaler
        self.scaler = StandardScaler()

        # Final model will be selected after comparison
        self.model = None

        # Store model comparison results
        self.model_results = {}

        # Store name of selected model
        self.best_model_name = None

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def prepare_data(self, data):

        try:

            self.logger.info("Preparing data for model training")

            # Separate features and target
            X = data.drop("target", axis=1)
            y = data["target"]

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # Fit scaler only on training data
            X_train = self.scaler.fit_transform(X_train)

            # Transform test data using training scaler
            X_test = self.scaler.transform(X_test)

            self.logger.info(
                f"Training data shape: {X_train.shape}"
            )

            self.logger.info(
                f"Testing data shape: {X_test.shape}"
            )

            self.logger.info(
                "Data preprocessing completed successfully"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:

            self.logger.error(
                f"Error during data preparation: {e}"
            )

            raise

    # =========================================================
    # CREATE MODELS
    # =========================================================

    def create_models(self):

        models = {

            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),

            "SVM": SVC(
                probability=True,
                random_state=42
            ),

            "KNN": KNeighborsClassifier(
                n_neighbors=5
            )
        }

        return models

    # =========================================================
    # MODEL COMPARISON
    # =========================================================

    def compare_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            print("\n")
            print("=" * 70)
            print("MODEL COMPARISON USING ROC-AUC")
            print("=" * 70)

            self.logger.info(
                "Starting machine learning model comparison"
            )

            models = self.create_models()

            # 5-Fold Stratified Cross Validation
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            # Store ROC curve values
            roc_curves = {}

            for name, model in models.items():

                print(f"\nEvaluating: {name}")

                # -------------------------------------------------
                # 5-FOLD CROSS VALIDATION
                # -------------------------------------------------

                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc"
                )

                mean_cv_auc = cv_scores.mean()
                std_cv_auc = cv_scores.std()

                # -------------------------------------------------
                # TRAIN MODEL ON TRAINING DATA
                # -------------------------------------------------

                model.fit(
                    X_train,
                    y_train
                )

                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                predictions = model.predict(
                    X_test
                )

                probabilities = model.predict_proba(
                    X_test
                )[:, 1]

                # -------------------------------------------------
                # ACCURACY
                # -------------------------------------------------

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                # -------------------------------------------------
                # TEST ROC-AUC
                # -------------------------------------------------

                test_auc = roc_auc_score(
                    y_test,
                    probabilities
                )

                # -------------------------------------------------
                # ROC CURVE
                # -------------------------------------------------

                fpr, tpr, thresholds = roc_curve(
                    y_test,
                    probabilities
                )

                roc_curves[name] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": test_auc
                }

                # -------------------------------------------------
                # STORE RESULTS
                # -------------------------------------------------

                self.model_results[name] = {
                    "accuracy": accuracy,
                    "test_auc": test_auc,
                    "cv_mean_auc": mean_cv_auc,
                    "cv_std_auc": std_cv_auc
                }

                # -------------------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------------------

                print(
                    f"Accuracy           : {accuracy:.4f}"
                )

                print(
                    f"Test ROC-AUC       : {test_auc:.4f}"
                )

                print(
                    f"5-Fold Mean AUC    : {mean_cv_auc:.4f}"
                )

                print(
                    f"5-Fold Std AUC     : {std_cv_auc:.4f}"
                )

                # Logging
                self.logger.info(
                    f"{name} | "
                    f"Accuracy={accuracy:.4f} | "
                    f"Test AUC={test_auc:.4f} | "
                    f"CV Mean AUC={mean_cv_auc:.4f} | "
                    f"CV Std={std_cv_auc:.4f}"
                )

            # =====================================================
            # PRINT COMPARISON TABLE
            # =====================================================

            print("\n")
            print("=" * 85)
            print("FINAL MODEL COMPARISON")
            print("=" * 85)

            print(
                f"{'Model':<25}"
                f"{'Accuracy':<15}"
                f"{'Test AUC':<15}"
                f"{'CV Mean AUC':<15}"
                f"{'CV Std':<10}"
            )

            print("-" * 85)

            for name, result in self.model_results.items():

                print(
                    f"{name:<25}"
                    f"{result['accuracy']:<15.4f}"
                    f"{result['test_auc']:<15.4f}"
                    f"{result['cv_mean_auc']:<15.4f}"
                    f"{result['cv_std_auc']:<10.4f}"
                )

            # =====================================================
            # SELECT BEST MODEL
            # =====================================================

            self.best_model_name = max(
                self.model_results,
                key=lambda name:
                self.model_results[name]["cv_mean_auc"]
            )

            print("\n")
            print("=" * 70)
            print("BEST MODEL SELECTION")
            print("=" * 70)

            print(
                f"Selected Model : {self.best_model_name}"
            )

            print(
                f"Mean CV AUC    : "
                f"{self.model_results[self.best_model_name]['cv_mean_auc']:.4f}"
            )

            print(
                f"Test AUC       : "
                f"{self.model_results[self.best_model_name]['test_auc']:.4f}"
            )

            print(
                f"Accuracy       : "
                f"{self.model_results[self.best_model_name]['accuracy']:.4f}"
            )

            self.logger.info(
                f"Best model selected: {self.best_model_name}"
            )

            # =====================================================
            # SET BEST MODEL
            # =====================================================

            self.model = models[self.best_model_name]

            # Train selected model
            self.model.fit(
                X_train,
                y_train
            )

            # =====================================================
            # PLOT ROC CURVES
            # =====================================================

            self.plot_roc_curves(
                roc_curves
            )

            return self.best_model_name

        except Exception as e:

            self.logger.error(
                f"Error during model comparison: {e}"
            )

            raise

    # =========================================================
    # ROC CURVE PLOT
    # =========================================================

    def plot_roc_curves(self, roc_curves):

        try:

            # Create results folder
            os.makedirs(
                "results",
                exist_ok=True
            )

            plt.figure(
                figsize=(10, 7)
            )

            # Plot each model
            for name, values in roc_curves.items():

                fpr = values["fpr"]
                tpr = values["tpr"]
                auc_score = values["auc"]

                plt.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=f"{name} (AUC = {auc_score:.3f})"
                )

            # Random classifier
            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                linewidth=1.5,
                label="Random Classifier (AUC = 0.500)"
            )

            plt.xlabel(
                "False Positive Rate"
            )

            plt.ylabel(
                "True Positive Rate"
            )

            plt.title(
                "ROC Curve Comparison - Heart Disease Prediction"
            )

            plt.legend(
                loc="lower right"
            )

            plt.grid(
                alpha=0.3
            )

            plt.tight_layout()

            output_path = (
                "results/roc_curve_comparison.png"
            )

            plt.savefig(
                output_path,
                dpi=300
            )

            plt.close()

            print(
                f"\nROC curve saved to: {output_path}"
            )

            self.logger.info(
                f"ROC curve saved to: {output_path}"
            )

        except Exception as e:

            self.logger.error(
                f"Error while creating ROC curve: {e}"
            )

            raise

    # =========================================================
    # TRAIN FINAL MODEL
    # =========================================================

    def train_model(
        self,
        X_train,
        y_train
    ):

        try:

            if self.model is None:

                raise ValueError(
                    "No model selected. "
                    "Run compare_models() first."
                )

            self.logger.info(
                f"Training final model: "
                f"{self.best_model_name}"
            )

            self.model.fit(
                X_train,
                y_train
            )

            print("\n")
            print(
                f"Final model trained: "
                f"{self.best_model_name}"
            )

            self.logger.info(
                "Final model training completed successfully"
            )

        except Exception as e:

            self.logger.error(
                f"Error during final model training: {e}"
            )

            raise

    # =========================================================
    # FINAL MODEL EVALUATION
    # =========================================================

    def evaluate_model(
        self,
        X_test,
        y_test
    ):

        try:

            # Predictions
            predictions = self.model.predict(
                X_test
            )

            # Probability of class 1
            probabilities = self.model.predict_proba(
                X_test
            )[:, 1]

            # Accuracy
            accuracy = accuracy_score(
                y_test,
                predictions
            )

            # ROC-AUC
            auc_score = roc_auc_score(
                y_test,
                probabilities
            )

            # Classification report
            report = classification_report(
                y_test,
                predictions
            )

            # Confusion matrix
            matrix = confusion_matrix(
                y_test,
                predictions
            )

            print("\n")
            print("=" * 70)
            print("FINAL MODEL EVALUATION")
            print("=" * 70)

            print(
                f"\nModel: {self.best_model_name}"
            )

            print(
                f"Accuracy: {accuracy:.4f}"
            )

            print(
                f"ROC-AUC: {auc_score:.4f}"
            )

            print(
                "\nClassification Report:"
            )

            print(
                report
            )

            print(
                "Confusion Matrix:"
            )

            print(
                matrix
            )

            self.logger.info(
                f"Final model evaluation completed | "
                f"Model={self.best_model_name} | "
                f"Accuracy={accuracy:.4f} | "
                f"ROC-AUC={auc_score:.4f}"
            )

            return accuracy, auc_score

        except Exception as e:

            self.logger.error(
                f"Error during model evaluation: {e}"
            )

            raise

    # =========================================================
    # SAVE MODEL AND SCALER
    # =========================================================

    def save_model(self):

        try:

            # Create models folder
            os.makedirs(
                "models",
                exist_ok=True
            )

            model_path = (
                "models/heart_model.pkl"
            )

            scaler_path = (
                "models/scaler.pkl"
            )

            # Save model
            joblib.dump(
                self.model,
                model_path
            )

            # Save scaler
            joblib.dump(
                self.scaler,
                scaler_path
            )

            self.logger.info(
                "Final model and scaler saved successfully"
            )

            print("\n")
            print("=" * 50)
            print("MODEL SAVING")
            print("=" * 50)

            print(
                f"Model : {model_path}"
            )

            print(
                f"Scaler: {scaler_path}"
            )

        except Exception as e:

            self.logger.error(
                f"Error while saving model: {e}"
            )

            raise
