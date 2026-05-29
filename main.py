import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("student_data.csv")

# Converting Exam_Score to binary classification (Pass/Fail) for Logistic Regression

data["Performance"] = data["Exam_Score"].apply(
    lambda x: 1 if x >= 70 else 0
)
data.drop("Exam_Score", axis=1, inplace=True)

# Exploratory Data Analysis
print(f"\nData Head:\n {data.head()}")
print(f"\nMissing Values:\n {data.isnull().sum()}")
print(f"\nData Description:\n {data.describe()}")
print(f"\nDuplicated Values: {data.duplicated().sum()}")

# Missing value fill
data["Teacher_Quality"]= data["Teacher_Quality"].fillna(data["Teacher_Quality"].mode()[0])
data["Parental_Education_Level"]= data["Parental_Education_Level"].fillna(data["Parental_Education_Level"].mode()[0])
data["Distance_from_Home"]= data["Distance_from_Home"].fillna(data["Distance_from_Home"].mode()[0])

print(f"\nAfter Filling Missing Values:\n {data.isnull().sum()}")

# Non Numeric Columns
categorical_cols = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities", 
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality", 
    "School_Type", "Peer_Influence", "Learning_Disabilities", "Parental_Education_Level", 
    "Distance_from_Home", "Gender"
]

le = LabelEncoder()

data[categorical_cols] = data[categorical_cols].apply(le.fit_transform)
print(f"\nTransformed Data:\n {data.head()}")

scale = StandardScaler()

X_scaled = scale.fit_transform(data.drop("Performance", axis=1))
y = data["Performance"]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LOGISTIC REGRESSION ( BASIC )

lr = LogisticRegression()

lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
print("\nLogistic Regression\n")
print(f"Logistic Regression Predicted Performance: {prediction[0]}")

accuracy_lr = accuracy_score(y_test, prediction)
precision_lr = precision_score(y_test, prediction)
roc_lr = roc_auc_score(y_test, prediction)
recall_lr = recall_score(y_test, prediction)

print("Accuracy:", f"{accuracy_lr:.2f}")
print("ROC AUC Score:", f"{roc_lr:.2f}")
print("Recall:", f"{recall_lr:.2f}")
print("Precision:", f"{precision_lr:.2f}")
print("\n-----------------------------\n")

# DECISION TREE CLASSIFIER ( INTERMEDIATE )  

dt = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10)

dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)
print("Decision Tree\n")
print(f"Decision Tree Predicted Performance: {dt_prediction[0]}")

dt_accuracy = accuracy_score(y_test, dt_prediction)
dt_roc = roc_auc_score(y_test, dt_prediction)
dt_recall = recall_score(y_test, dt_prediction)
dt_precision = precision_score(y_test, dt_prediction)
print("Accuracy:", f"{dt_accuracy:.2f}")
print("ROC AUC Score:", f"{dt_roc:.2f}")
print("Recall:", f"{dt_recall:.2f}")
print("Precision:", f"{dt_precision:.2f}")
