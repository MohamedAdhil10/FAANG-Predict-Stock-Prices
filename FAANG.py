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

#1. Data Cleaning
# Load dataset
df = pd.read_csv("FAANG.csv")

df['Date'] = pd.to_datetime(df['Date'])

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

filtered_df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Scale numerical data
scaler = StandardScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

#2. Exploratory Data Analysis
# Line Chart: Stock Price Trends Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label="Stock Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("FAANG Stock Price Trend Over Time")
plt.legend()
plt.grid()
plt.show()

## Correlation Heatmap: Identifying Relationships Between Features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Box Plot: Detecting Outliers in Prices & Volume
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Open', 'High', 'Low', 'Close', 'Volume']], palette='coolwarm')
plt.title("Box Plot of Stock Prices & Volume")
plt.xticks(rotation=45)
plt.show()

# Scatter Plot: Relationship Between Volume & Closing Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Volume'], y=df['Close'], alpha=0.5, color='green')
plt.xlabel("Trading Volume")
plt.ylabel("Closing Price")
plt.title("Relationship Between Volume & Closing Price")
plt.show()


# 3. Model Development
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)

# Random Search (optional for faster tuning)
random_grid = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 11, 2)
}

print("Running Random Search...")
random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)
best_random_model = random_search.best_estimator_

# 4. Model Evaluation
y_pred = best_random_model.predict(X_test)

# Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save Model
joblib.dump(best_random_model, "faang_model.pkl")

# 5. MLflow Intergration
mlflow.set_experiment("FAANG Stock Price Prediction")
with mlflow.start_run():
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(best_random_model, "model")

# 6. Streamlit

# Cache model & dataset to prevent reloading
@st.cache_resource()
def load_model():
    return joblib.load("faang_model.pkl")

@st.cache_data()
def load_data():
    return pd.read_csv("FAANG.csv")

# Load model and dataset
model = load_model()
df = load_data()

# Streamlit UI
st.title("FAANG Stock Price Prediction")

# User Inputs
open_price = st.number_input("Enter Open Price", value=0.5)
high_price = st.number_input("Enter High Price", value=0.5)
low_price = st.number_input("Enter Low Price", value=0.5)
volume = st.number_input("Enter Volume", value=0.5)

# Function for prediction (cached to avoid recomputation)
@st.cache_resource()
def predict(open_price, high_price, low_price, volume):
    return model.predict([[open_price, high_price, low_price, volume]])[0]

if st.button("Predict Closing Price"):
    with st.spinner("Predicting..."):
        prediction = predict(open_price, high_price, low_price, volume)
        st.success(f"Predicted Closing Price: **${prediction:.2f}**")
