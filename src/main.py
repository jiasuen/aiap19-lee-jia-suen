# -*- coding: utf-8 -*-
"""Main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1K0hd7zlCPyikvaWdQq1tITh0LVRyzEvZ
"""

import shutil
import os
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from preprocessing import preprocess_data  # Import the preprocessing function
from data_columns import categorical_columns
from models import (
    prepare_data,
    train_rf_task1, train_rf_task2,
    train_nn_task1, train_nn_task2,
    train_xgb_task1, train_xgb_task2
)


destination = "/Users/jiasuen/Documents/Y4S1/AIAP/data/agri.db"

# Connect to the SQLite database
connection = sqlite3.connect(destination)


cursor = connection.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)


table_name = "farm_data"
plants_df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


print("Data loaded from SQLite database:")
print(plants_df.head())


connection.close()

# Step 2: Preprocess the dataset
plants_df = preprocess_data(plants_df)

# Step 3: Prepare data for both tasks
(
    X_task1_train, X_task1_test, y_task1_train, y_task1_test,
    X_task2_train, X_task2_test, y_task2_train, y_task2_test,
    label_encoders, label_encoder_target
) = prepare_data(plants_df, categorical_columns)

# Step 4: Train and evaluate models
# Task 1 - Random Forest
rf_model_task1, mse_rf, rmse_rf, r2_rf = train_rf_task1(X_task1_train, y_task1_train, X_task1_test, y_task1_test)
print("Task 1 (Random Forest) Results:")
print(f"MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# Task 2 - Random Forest
rf_model_task2, accuracy_rf, report_rf, matrix_rf = train_rf_task2(
    X_task2_train, y_task2_train, X_task2_test, y_task2_test, label_encoder_target
)
print("Task 2 (Random Forest) Results:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(report_rf)

# Task 1 - Neural Network
nn_model_task1, history_nn_task1, test_loss_nn, test_mae_nn = train_nn_task1(X_task1_train, y_task1_train, X_task1_test, y_task1_test)
print("Task 1 (Neural Network) Results:")
print(f"Test Loss: {test_loss_nn:.4f}, MAE: {test_mae_nn:.4f}")

# Task 2 - Neural Network
nn_model_task2, history_nn_task2, test_loss_nn_task2, test_accuracy_nn = train_nn_task2(
    X_task2_train, y_task2_train, X_task2_test, y_task2_test, num_classes=len(label_encoder_target.classes_)
)
print("Task 2 (Neural Network) Results:")
print(f"Test Loss: {test_loss_nn_task2:.4f}, Accuracy: {test_accuracy_nn:.4f}")

# Task 1 - XGBoost
xgb_model_task1, mse_xgb, rmse_xgb, r2_xgb = train_xgb_task1(X_task1_train, y_task1_train, X_task1_test, y_task1_test)
print("Task 1 (XGBoost) Results:")
print(f"MSE: {mse_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

# Task 2 - XGBoost
xgb_model_task2, accuracy_xgb, report_xgb, matrix_xgb = train_xgb_task2(
    X_task2_train, y_task2_train, X_task2_test, y_task2_test, label_encoder_target
)
print("Task 2 (XGBoost) Results:")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(report_xgb)

# Step 5: Display summary results for all models
print("\nSummary of Results:")
print("Task 1 (Random Forest) - MSE:", mse_rf, "RMSE:", rmse_rf, "R²:", r2_rf)
print("Task 2 (Random Forest) - Accuracy:", accuracy_rf)
print("Task 1 (XGBoost) - MSE:", mse_xgb, "RMSE:", rmse_xgb, "R²:", r2_xgb)
print("Task 2 (XGBoost) - Accuracy:", accuracy_xgb)
print("Task 1 (Neural Network) - Loss:", test_loss_nn, "MAE:", test_mae_nn)
print("Task 2 (Neural Network) - Loss:", test_loss_nn_task2, "Accuracy:", test_accuracy_nn)
