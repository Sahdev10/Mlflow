import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Create a simple dataset
X = np.random.rand(100, 1) * 10  # Features
y = 2.5 * X.flatten() + np.random.randn(100) * 2  # Labels with noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Simple-Linear-Regression")

with mlflow.start_run():
    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters
    mlflow.log_param("fit_intercept", model.fit_intercept)

    # Log metrics
    mlflow.log_metric("mse", mse)

    # Log model
    mlflow.sklearn.log_model(model, "linear_model")

    print(f"Logged MSE: {mse}")
    print("Model saved in MLflow")

