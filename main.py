from src.data_preprocessing import DataPreprocessor
from src.model_training import HeartDiseaseModel


def main():

    print("\n")
    print("=" * 70)
    print("HEART DISEASE PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # =========================================================
    # STEP 1: LOAD DATASET
    # =========================================================

    file_path = "data/heart.csv"

    print("\n[1] Loading dataset...")

    preprocessor = DataPreprocessor(
        file_path
    )

    data = preprocessor.load_data()

    # Stop if dataset cannot be loaded
    if data is None:

        print(
            "\nUnable to load dataset."
        )

        return

    # =========================================================
    # STEP 2: CHECK DATASET
    # =========================================================

    print("\n[2] Checking dataset...")

    preprocessor.check_data(
        data
    )

    # =========================================================
    # STEP 3: CREATE MODEL OBJECT
    # =========================================================

    print("\n[3] Creating machine learning pipeline...")

    heart_model = HeartDiseaseModel()

    # =========================================================
    # STEP 4: PREPARE DATA
    # =========================================================

    print("\n[4] Preparing data...")

    X_train, X_test, y_train, y_test = (
        heart_model.prepare_data(data)
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # =========================================================
    # STEP 5: COMPARE MACHINE LEARNING MODELS
    # =========================================================

    print(
        "\n[5] Comparing machine learning algorithms..."
    )

    best_model = heart_model.compare_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        f"\nSelected model: {best_model}"
    )

    # =========================================================
    # STEP 6: TRAIN FINAL MODEL
    # =========================================================

    print(
        "\n[6] Training final model..."
    )

    heart_model.train_model(
        X_train,
        y_train
    )

    # =========================================================
    # STEP 7: EVALUATE FINAL MODEL
    # =========================================================

    print(
        "\n[7] Evaluating final model..."
    )

    accuracy, auc_score = (
        heart_model.evaluate_model(
            X_test,
            y_test
        )
    )

    # =========================================================
    # STEP 8: SAVE MODEL
    # =========================================================

    print(
        "\n[8] Saving trained model..."
    )

    heart_model.save_model()

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nFinal Model      : {best_model}"
    )

    print(
        f"Test Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Test ROC-AUC     : {auc_score:.4f}"
    )

    print(
        "\nGenerated Files:"
    )

    print(
        "  models/heart_model.pkl"
    )

    print(
        "  models/scaler.pkl"
    )

    print(
        "  results/roc_curve_comparison.png"
    )

    print(
        "\nThe project is ready for prediction testing."
    )


# =============================================================
# PROGRAM ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()