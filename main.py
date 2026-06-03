import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,roc_auc_score,recall_score,precision_score,confusion_matrix,f1_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = pd.read_csv("student_data.csv")

# Conversion of Exam_Score to binary Performance ( Regression to Classification )
data["Performance"] = data["Exam_Score"].apply( lambda x: 1 if x >= 70 else 0 )

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LOGISTIC REGRESSION

lr = LogisticRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

print("\n===== Logistic Regression =====")

print("Before Regularization:")

print(f"Accuracy: {accuracy_score(y_test, prediction):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, prediction):.2f}")
print(f"Recall: {recall_score(y_test, prediction):.2f}")
print(f"Precision: {precision_score(y_test, prediction):.2f}")

print("\nAfter L2 Regularization:")

lr_l2 = LogisticRegression(penalty='l2', C=0.1, max_iter=800, random_state=42)
lr_l2.fit(X_train, y_train)
l2_prediction = lr_l2.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, l2_prediction):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, l2_prediction):.2f}")
print(f"Recall: {recall_score(y_test, l2_prediction):.2f}")
print(f"Precision: {precision_score(y_test, l2_prediction):.2f}")

print("\nAfter L1 Regularization:")

lr_l1 = LogisticRegression(penalty='l1', C=0.1, random_state=42, solver='liblinear')
lr_l1.fit(X_train, y_train)
l1_prediction = lr_l1.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, l1_prediction):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, l1_prediction):.2f}")
print(f"Recall: {recall_score(y_test, l1_prediction):.2f}")
print(f"Precision: {precision_score(y_test, l1_prediction):.2f}")


# DECISION TREE
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_prediction = dt.predict(X_test)

print("\n===== Decision Tree =====")

print("Before Regularization:")

print(f"Accuracy: {accuracy_score(y_test, dt_prediction):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, dt_prediction):.2f}")
print(f"Recall: {recall_score(y_test, dt_prediction):.2f}")
print(f"Precision: {precision_score(y_test, dt_prediction):.2f}")

print("\nAfter Regularization")

dt_reg = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=10,random_state=42)
dt_reg.fit(X_train, y_train)
dt_reg_prediction = dt_reg.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, dt_reg_prediction):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, dt_reg_prediction):.2f}")
print(f"Recall: {recall_score(y_test, dt_reg_prediction):.2f}")
print(f"Precision: {precision_score(y_test, dt_reg_prediction):.2f}")

#XGBOOST ( ADVANCED )

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_prediction = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1] 

tn, fp, fn, tp = confusion_matrix(y_test, xgb_prediction).ravel()
xgb_specificity = tn / (tn + fp)
xgb_f1 = f1_score(y_test, xgb_prediction)
loss_xg = log_loss(y_test, xgb_prob)

print("\n----------BEFORE REGULARIZATION--------------\n")
print(f"Loss XGBoost: {loss_xg}\n")
print("XGBoost Classifier (Advanced Model)\n")
print(f"Accuracy:    {accuracy_score(y_test, xgb_prediction):.2f}")
print(f"ROC AUC:     {roc_auc_score(y_test, xgb_prob):.2f}")
print(f"Precision:   {precision_score(y_test, xgb_prediction):.2f}")
print(f"Recall:      {recall_score(y_test, xgb_prediction):.2f}")
print(f"F1-Score:    {xgb_f1:.2f}")
print(f"Specificity: {xgb_specificity:.2f}")

print("\n----------AFTER REGULARIZATION--------------\n")

xgb_reg = xgb.XGBClassifier(
    random_state=42, 
    eval_metric='logloss', 
    reg_lambda=10, # for L2 regularization ( penalize weights )
    reg_alpha=10,  # for L1 regularization (removes less important features)
    max_depth=3,
    learning_rate=0.3,
    )
xgb_reg.fit(X_train, y_train)
xgb_reg_prediction = xgb_reg.predict(X_test)
xgb_reg_prob = xgb_reg.predict_proba(X_test)[:, 1]
loss_xg_reg = log_loss(y_test, xgb_reg_prob)

print(f"Loss XGBoost (Regularized): {loss_xg_reg}\n")
print("XGBoost Classifier (Regularized) (Advanced Model)\n")
print(f"Accuracy:    {accuracy_score(y_test, xgb_reg_prediction):.2f}")
print(f"ROC AUC:     {roc_auc_score(y_test, xgb_reg_prob):.2f}")
print(f"Precision:   {precision_score(y_test, xgb_reg_prediction):.2f}")
print(f"Recall:      {recall_score(y_test, xgb_reg_prediction):.2f}")
print(f"F1-Score:    {f1_score(y_test, xgb_reg_prediction):.2f}")

#print(f"Specificity: {silhouette_score(X_scaled, kmeans.labels_):.2f}")

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