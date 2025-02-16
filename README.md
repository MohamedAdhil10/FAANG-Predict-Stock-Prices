# FAANG Stock Price Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-success?style=for-the-badge&logo=streamlit) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow) ![scikit-learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-orange?style=for-the-badge&logo=scikit-learn)  

## About the Project

This project **predicts the closing stock price** of FAANG stocks (Facebook, Apple, Amazon, Netflix, Google) using **Machine Learning & Streamlit**.  
It trains a **RandomForestRegressor** model on historical stock data and allows users to input stock prices & volume to predict closing prices.  

**Tech Stack:** Python, Pandas, Scikit-Learn, Seaborn, Matplotlib  
**Deployed on:** Streamlit  
**Model Logging & Tracking:** MLflow  

---

## Features

**Data Cleaning & Preprocessing**  
**Exploratory Data Analysis (EDA)** (Line Chart, Correlation Heatmap, Box Plot, Scatter Plot)  
**Machine Learning Model (RandomForestRegressor)**  
**Hyperparameter Tuning (Random Search + Grid Search)**  
**Streamlit Web App for User Predictions**  
**MLflow Integration for Model Tracking**  
**Deployed Online via Streamlit Cloud**  

---

## Data Pipeline

1 **Data Cleaning** → Handles missing values, removes outliers, standardizes features  
2 **Exploratory Data Analysis (EDA)** → Generates stock trends, correlation heatmap  
3 **Feature Selection** → Uses `Open`, `High`, `Low`, `Volume` to predict `Close` price  
4 **Model Development** → Trains a **RandomForestRegressor**  
5 **Hyperparameter Tuning** → Uses **RandomizedSearchCV** to optimize parameters  
6 **Model Evaluation** → Computes **MAE, RMSE, R² Score**  
7 **Deployment** → Streamlit app for live stock predictions  

---

## Installation Guide

### **1 Clone the Repository**
```bash
git clone https://github.com/your-username/FAANG-Stock-Prediction.git
cd FAANG-Stock-Prediction
```

### **2 Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3 Run the Streamlit App**
```bash
streamlit run FAANG.py
```

---

## Project Structure

```
FAANG-Stock-Prediction
│-- 📄 FAANG.py                # Main Streamlit app
│-- 📄 FAANG.csv               # Stock dataset
|-- 📄 requirements.txt        # Python dependencies
│-- 📄 README.md               # Project documentation

```

---

## **Model Performance**

| Metric | Score |
|--------|-------|
| **MAE** | `0.0049` |
| **RMSE** | `0.0126` |
| **R² Score** | `0.9998` |

---

## **How to Use the App?**
1 **Enter stock values**: `Open Price`, `High Price`, `Low Price`, `Volume`  
2 **Click "Predict Closing Price"**  
3 **View the predicted closing stock price**  

---
## **Streamlit App Screenshot**
![FAANG](https://github.com/user-attachments/assets/0c989b37-9ffd-464f-849a-7a5a1a841e8c)


---

## **MLflow Experiment Tracking**
This project uses **MLflow** to log model performance, hyperparameters, and track experiments.  

**Tracking MAE, RMSE, R²**  
**Logging the best model from Randomized Search**  
**Saving the trained model for deployment**  

To run MLflow locally:
```bash
mlflow ui
```
Then, open **`http://192.168.1.6:8501`** in your browser.

---


## **Contact**
**Email:** adhilm9991@gmail.com  
**GitHub:** https://github.com/MohamedAdhil10 
**LinkedIn:** https://www.linkedin.com/in/mohamed-adhil-99118b247 

---

