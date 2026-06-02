import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,roc_auc_score,recall_score,precision_score,confusion_matrix,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = pd.read_csv("student_data.csv")

# Conversion of Exam_Score to binary Performance ( Regression to Classification )
data["Performance"] = data["Exam_Score"].apply(
    lambda x: 1 if x >= 70 else 0
)

data.drop("Exam_Score", axis=1, inplace=True)


# EDA

print("\nData Head:\n", data.head())
print("\nMissing Values:\n", data.isnull().sum())
print("\nDescription:\n", data.describe())
print("\nDuplicate Values:", data.duplicated().sum())


# Missing Value fill with mode for categorical features

data["Teacher_Quality"] = data["Teacher_Quality"].fillna(data["Teacher_Quality"].mode()[0])
data["Parental_Education_Level"] = data[ "Parental_Education_Level"].fillna(data["Parental_Education_Level"].mode()[0])
data["Distance_from_Home"] = data["Distance_from_Home"].fillna(data["Distance_from_Home"].mode()[0])

print("\nAfter Filling Missing Values:\n")
print(data.isnull().sum())

# One Hot Encoding for Categorical Variables

categorical_cols = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

data = pd.get_dummies(
    data,
    columns=categorical_cols,
    drop_first=True
)

print("\nEncoded Data Shape:", data.shape)
print("\nEncoded Data Head:\n", data.head())


X = data.drop("Performance", axis=1)
y = data["Performance"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# LOGISTIC REGRESSION

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

print("\n===== Logistic Regression =====")

print("Prediction:", prediction[0])
print("Accuracy:", round(accuracy_score(y_test, prediction), 2))
print("ROC AUC:", round(roc_auc_score(y_test, prediction), 2))
print("Recall:", round(recall_score(y_test, prediction), 2))
print("Precision:", round(precision_score(y_test, prediction), 2))

# ── AFTER Regularization: Logistic Regression ──

print("\n===== AFTER Regularization =====")

# L2 - shrinks all weights, keeps all features
lr_l2 = LogisticRegression(max_iter=1000, penalty="l2", C=0.1)
lr_l2.fit(X_train, y_train)
pred_l2 = lr_l2.predict(X_test)
print("\n-- Logistic Regression: L2 (C=0.1) --")
print("Train Accuracy:", round(accuracy_score(y_train, lr_l2.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_l2), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_l2), 2))

# L1 - removes weak/useless features by pushing their weights to zero
lr_l1 = LogisticRegression(max_iter=1000, penalty="l1", C=0.1, solver="liblinear")
lr_l1.fit(X_train, y_train)
pred_l1 = lr_l1.predict(X_test)
print("\n-- Logistic Regression: L1 (C=0.1) --")
print("Train Accuracy:", round(accuracy_score(y_train, lr_l1.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_l1), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_l1), 2))

# ElasticNet - combination of both L1 and L2
lr_en = LogisticRegression(max_iter=1000, penalty="elasticnet", C=0.1, solver="saga", l1_ratio=0.5)
lr_en.fit(X_train, y_train)
pred_en = lr_en.predict(X_test)
print("\n-- Logistic Regression: ElasticNet (C=0.1) --")
print("Train Accuracy:", round(accuracy_score(y_train, lr_en.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_en), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_en), 2))

# DECISION TREE

dt = DecisionTreeClassifier(
    max_depth=5,
    max_leaf_nodes=10,
    random_state=42
)

dt.fit(X_train, y_train)

dt_prediction = dt.predict(X_test)

print("\n===== Decision Tree =====")

print("Prediction:", dt_prediction[0])
print("Accuracy:", round(accuracy_score(y_test, dt_prediction), 2))
print("ROC AUC:", round(roc_auc_score(y_test, dt_prediction), 2))
print("Recall:", round(recall_score(y_test, dt_prediction), 2))
print("Precision:", round(precision_score(y_test, dt_prediction), 2))

# ── AFTER Regularization: Decision Tree ──

# Shallow tree - limiting depth keeps model simple
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)
pred_shallow = dt_shallow.predict(X_test)
print("\n-- Decision Tree: Shallow (max_depth=3) --")
print("Train Accuracy:", round(accuracy_score(y_train, dt_shallow.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_shallow), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_shallow), 2))

# Min samples leaf - each leaf must have at least 20 samples
dt_min = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
dt_min.fit(X_train, y_train)
pred_min = dt_min.predict(X_test)
print("\n-- Decision Tree: min_samples_leaf=20 --")
print("Train Accuracy:", round(accuracy_score(y_train, dt_min.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_min), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_min), 2))

# Pruning - removes branches that don't improve accuracy much
dt_pruned = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
dt_pruned.fit(X_train, y_train)
pred_pruned = dt_pruned.predict(X_test)
print("\n-- Decision Tree: Pruned (ccp_alpha=0.01) --")
print("Train Accuracy:", round(accuracy_score(y_train, dt_pruned.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_pruned), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, pred_pruned), 2))

#XGBOOST ( ADVANCED )

xgb_model = xgb.XGBClassifier(
    random_state=42, eval_metric='logloss', reg_lambda=1.0, max_depth=5
)
xgb_model.fit(X_train, y_train)

xgb_prediction = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1] 

tn, fp, fn, tp = confusion_matrix(y_test, xgb_prediction).ravel()
xgb_specificity = tn / (tn + fp)
xgb_f1 = f1_score(y_test, xgb_prediction)

print("\n-----------------------------\n")
print("XGBoost Classifier (Advanced Model)\n")
print(f"Accuracy:    {accuracy_score(y_test, xgb_prediction):.2f}")
print(f"ROC AUC:     {roc_auc_score(y_test, xgb_prob):.2f}")
print(f"Precision:   {precision_score(y_test, xgb_prediction):.2f}")
print(f"Recall:      {recall_score(y_test, xgb_prediction):.2f}")
print(f"F1-Score:    {xgb_f1:.2f}")
print(f"Specificity: {xgb_specificity:.2f}")

# ── AFTER Regularization: XGBoost ──

# L2 only - stronger than the default (reg_lambda=1.0)
xgb_l2 = xgb.XGBClassifier(random_state=42, eval_metric='logloss', reg_lambda=5.0, reg_alpha=0, max_depth=5)
xgb_l2.fit(X_train, y_train)
pred_xgb_l2 = xgb_l2.predict(X_test)
print("\n-- XGBoost: L2 (reg_lambda=5.0) --")
print("Train Accuracy:", round(accuracy_score(y_train, xgb_l2.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_xgb_l2), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, xgb_l2.predict_proba(X_test)[:,1]), 2))

# L1 only - removes less useful features
xgb_l1 = xgb.XGBClassifier(random_state=42, eval_metric='logloss', reg_lambda=0, reg_alpha=1.0, max_depth=5)
xgb_l1.fit(X_train, y_train)
pred_xgb_l1 = xgb_l1.predict(X_test)
print("\n-- XGBoost: L1 (reg_alpha=1.0) --")
print("Train Accuracy:", round(accuracy_score(y_train, xgb_l1.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_xgb_l1), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, xgb_l1.predict_proba(X_test)[:,1]), 2))

# L1 + L2 combined
xgb_both = xgb.XGBClassifier(random_state=42, eval_metric='logloss', reg_lambda=2.0, reg_alpha=1.0, max_depth=5)
xgb_both.fit(X_train, y_train)
pred_xgb_both = xgb_both.predict(X_test)
print("\n-- XGBoost: L1 + L2 Combined --")
print("Train Accuracy:", round(accuracy_score(y_train, xgb_both.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_xgb_both), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, xgb_both.predict_proba(X_test)[:,1]), 2))

# Shallow + regularized
xgb_shallow = xgb.XGBClassifier(random_state=42, eval_metric='logloss', reg_lambda=2.0, reg_alpha=1.0, max_depth=3)
xgb_shallow.fit(X_train, y_train)
pred_xgb_shallow = xgb_shallow.predict(X_test)
print("\n-- XGBoost: Shallow + Regularized (max_depth=3) --")
print("Train Accuracy:", round(accuracy_score(y_train, xgb_shallow.predict(X_train)), 2))
print("Test Accuracy: ", round(accuracy_score(y_test, pred_xgb_shallow), 2))
print("Test ROC AUC:  ", round(roc_auc_score(y_test, xgb_shallow.predict_proba(X_test)[:,1]), 2))


# ELBOW METHOD ( To Know Correct Number of Clusters )

wcss = []

for i in range(1, 11):
    km = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

#  Unsupervised Clustering with K-Means

kmeans = KMeans(
    n_clusters=2,
    init='k-means++',
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

data["Cluster"] = kmeans.labels_

print("\nCluster Distribution:")
print(data["Cluster"].value_counts())

from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"\nSilhouette Score: {sil_score:.2f}")

# PCA (for visualization 20D to 2D)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

centroids_pca = pca.transform(
    kmeans.cluster_centers_
)


# KMEANS VISUALIZATION

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[data["Cluster"] == 0, 0],
    X_pca[data["Cluster"] == 0, 1],
    label="Cluster 0",
    alpha=0.6
)

plt.scatter(
    X_pca[data["Cluster"] == 1, 0],
    X_pca[data["Cluster"] == 1, 1],
    label="Cluster 1",
    alpha=0.6
)

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    s=250,
    marker='X',
    label="Centroids"
)

plt.title("K-Means Student Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()


# ACTUAL PERFORMANCE VIEW


plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=data["Performance"],
    cmap='coolwarm',
    alpha=0.7
)

plt.colorbar(label="Performance")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Actual Student Performance Distribution")
plt.grid(True)
plt.show()