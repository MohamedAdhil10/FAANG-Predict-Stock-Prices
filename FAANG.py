import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import streamlit as st
import joblib

# 1. Data Cleaning
# Load dataset
df = pd.read_csv("FAANG.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Drop empty and categorical columns
empty_columns = df.columns[df.isnull().all()].tolist()
df.drop(columns=empty_columns, inplace=True)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
df.drop(columns=categorical_columns, inplace=True)

# Handle missing values
num_imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Remove outliers using IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[~((df[numeric_cols] < (Q1 - 2.5 * IQR)) | (df[numeric_cols] > (Q3 + 2.5 * IQR))).any(axis=1)]

# 2. Feature Selection and Scaling
X = filtered_df[['Open', 'High', 'Low', 'Volume']]
y = filtered_df['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Development
rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)

# Random Search for Hyperparameter Tuning
random_grid = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 11, 2)
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
random_search.fit(X_train_scaled, y_train)
best_random_model = random_search.best_estimator_

# 4. Model Evaluation
y_pred = best_random_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save Model and Scaler
joblib.dump(best_random_model, "faang_random_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 5. MLflow Integration
mlflow.set_experiment("FAANG Stock Price Prediction")
with mlflow.start_run():
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(best_random_model, "model")

# 6. Streamlit Application
@st.cache_resource()
def load_model():
    return joblib.load("faang_random_model.pkl")

@st.cache_resource()
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

st.title("FAANG Stock Price Prediction")

open_price = st.number_input("Enter Open Price", value=0.5)
high_price = st.number_input("Enter High Price", value=0.5)
low_price = st.number_input("Enter Low Price", value=0.5)
volume = st.number_input("Enter Volume", value=0.5)

@st.cache_resource()
def predict(open_price, high_price, low_price, volume):
    scaled_input = scaler.transform([[open_price, high_price, low_price, volume]])
    return model.predict(scaled_input)[0]

if st.button("Predict Closing Price"):
    with st.spinner("Predicting..."):
        prediction = predict(open_price, high_price, low_price, volume)
        st.success(f"Predicted Closing Price: **${prediction:.2f}**")
