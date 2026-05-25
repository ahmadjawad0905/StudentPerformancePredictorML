import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("student_data.csv")

print(data.head())
print(data.isnull().sum())

# Missing value fill
data["Teacher_Quality"]= data["Teacher_Quality"].fillna(data["Teacher_Quality"].mode()[0])
data["Parental_Education_Level"]= data["Parental_Education_Level"].fillna(data["Parental_Education_Level"].mode()[0])
data["Distance_from_Home"]= data["Distance_from_Home"].fillna(data["Distance_from_Home"].mode()[0])

print(data.isnull().sum())

# Non Numeric Columns
categorical_cols = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities", 
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality", 
    "School_Type", "Peer_Influence", "Learning_Disabilities", "Parental_Education_Level", 
    "Distance_from_Home", "Gender"
]

le = LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(le.fit_transform)

print(data.head())


X = data.drop("Exam_Score", axis=1)
y = data["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LOGISTIC REGRESSION ( BASIC )

lr = LinearRegression()

lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
print("\nLinear Regression\n")
print(f"Linear Regression Predicted Exam Score: {prediction[0]:.2f}")

mse_lr = mean_squared_error(y_test, prediction)
r2_lr = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

print("Mean Squared Error:", mse_lr)
print("R2 Score:", r2_lr)
print("Mean Absolute Error:", mae)
print("\n-----------------------------\n")

# DECISION TREE REGRESSOR ( INTERMEDIATE )  

dt = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=10)

dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)
print("Decision Tree\n")
print(f"Decision Tree Predicted Exam Score: {dt_prediction[0]:.2f}")

dt_mse = mean_squared_error(y_test, dt_prediction)
dt_r2_score = r2_score(y_test, dt_prediction)
print("Decision Tree Mean Squared Error:", dt_mse)
print("Decision Tree R2 Score:", dt_r2_score)
