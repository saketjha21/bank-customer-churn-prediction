import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# Load the dataset
df = pd.read_csv("E:/Um-Project Sam/european-bank-churn-prediction/data/European_Bank.csv")

# Look at the first 5 rows
print(df.head())

# Check the shape (rows, columns)
print("Shape:", df.shape)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check how many customers churned vs stayed
print("\nChurn distribution:")
print(df["Exited"].value_counts())
print(df["Exited"].value_counts(normalize=True))

# Drop columns that don't help prediction
df_model = df.drop(columns=["CustomerId", "Surname", "Year"])
print("\nColumns after dropping unnecessary ones:")
print(df_model.columns.tolist())


# Create encoders for Geography and Gender
le_geo = LabelEncoder()
le_gender = LabelEncoder()

# Before encoding - see the original text values
print("\nGeography values before encoding:", df_model["Geography"].unique())
print("Gender values before encoding:", df_model["Gender"].unique())

# Apply encoding
df_model["Geography"] = le_geo.fit_transform(df_model["Geography"])
df_model["Gender"] = le_gender.fit_transform(df_model["Gender"])

# After encoding - see the numbers
print("\nGeography values after encoding:", df_model["Geography"].unique())
print("Gender values after encoding:", df_model["Gender"].unique())

# Feature Engineering - create new helpful columns
df_model["BalanceSalaryRatio"] = df_model["Balance"] / (df_model["EstimatedSalary"] + 1)
df_model["IsZeroBalance"] = (df_model["Balance"] == 0).astype(int)

print("\nNew engineered columns preview:")
print(df_model[["Balance", "EstimatedSalary", "BalanceSalaryRatio", "IsZeroBalance"]].head())


from sklearn.model_selection import train_test_split

# X = all the input features (everything except the answer)
# y = the target we want to predict (Exited)
X = df_model.drop(columns=["Exited"])
y = df_model["Exited"]

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("Churn rate in training set:", y_train.mean())
print("Churn rate in testing set:", y_test.mean())


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBefore scaling (first row):")
print(X_train.iloc[0].values)
print("\nAfter scaling (first row):")
print(X_train_scaled[0])



# Create and train the model
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)

# Use the trained model to predict on the test set (data it has never seen)
pred_lr = log_reg.predict(X_test_scaled)
proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate how good the predictions are
print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("Precision:", precision_score(y_test, pred_lr))
print("Recall:", recall_score(y_test, pred_lr))
print("F1 Score:", f1_score(y_test, pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, proba_lr))


from sklearn.ensemble import RandomForestClassifier

# Create and train the model (note: uses UNSCALED data)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
)
rf.fit(X_train, y_train)

# Predict on test set
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate
print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("Precision:", precision_score(y_test, pred_rf))
print("Recall:", recall_score(y_test, pred_rf))
print("F1 Score:", f1_score(y_test, pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, proba_rf))

# ------------------------------------------------------------------------------------
import joblib
import os

# Create a 'model' folder path 
model_dir = "E:/Um-Project Sam/european-bank-churn-prediction/model"
os.makedirs(model_dir, exist_ok=True)

# Save everything we'll need later in the Streamlit app
joblib.dump(rf, os.path.join(model_dir, "churn_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(le_geo, os.path.join(model_dir, "le_geo.pkl"))
joblib.dump(le_gender, os.path.join(model_dir, "le_gender.pkl"))

# Also save the exact column order the model expects
import json
with open(os.path.join(model_dir, "feature_order.json"), "w") as f:
    json.dump(list(X.columns), f)

print("\n✅ Model and all supporting files saved to the 'model' folder!")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Create outputs folder if it doesn't exist
outputs_dir = "E:/Um-Project Sam/european-bank-churn-prediction/outputs"
os.makedirs(outputs_dir, exist_ok=True)

sns.set_style("whitegrid")

# Chart 1: Churn distribution
plt.figure(figsize=(5, 4))
df["Exited"].value_counts().plot(kind="bar", color=["#2E7D32", "#C62828"])
plt.title("Customer Churn Distribution (0=Stayed, 1=Exited)")
plt.xlabel("Exited")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "01_churn_distribution.png"))
plt.close()

# Chart 2: Churn rate by Geography
plt.figure(figsize=(6, 4))
sns.barplot(x="Geography", y="Exited", data=df, estimator="mean")
plt.title("Churn Rate by Geography")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "02_churn_by_geography.png"))
plt.close()

print("\n✅ Charts saved in the 'outputs' folder!")

# Chart 3: Age distribution vs churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="Age", hue="Exited", fill=True, common_norm=False)
plt.title("Age Distribution: Churned vs Retained")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "03_age_vs_churn.png"))
plt.close()

# Chart 4: Correlation heatmap (numeric columns only)
plt.figure(figsize=(8, 6))
numeric_df = df_model.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "04_correlation_heatmap.png"))
plt.close()

# Chart 5: Confusion Matrix (for Random Forest - our best model)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Exited"], yticklabels=["Stayed", "Exited"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "05_confusion_matrix.png"))
plt.close()

# Chart 6: Feature importance (Random Forest)
plt.figure(figsize=(7, 5))
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh", color="#1565C0")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, "06_feature_importance.png"))
plt.close()

print("\n✅ All 6 charts saved in the 'outputs' folder!")