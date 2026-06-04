import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import sys
import dagshub
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read dataset
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "titanic_data_preprocessed.csv"
    )
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Survived", axis=1),
        data["Survived"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]

    # Hyperparameter
    solver = sys.argv[2] if len(sys.argv) > 2 else "liblinear"

    with mlflow.start_run():

        model = LogisticRegression(random_state=42, solver=solver)
        model.fit(X_train, y_train)

        predicted_survived = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        model.fit(X_train, y_train)

        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)