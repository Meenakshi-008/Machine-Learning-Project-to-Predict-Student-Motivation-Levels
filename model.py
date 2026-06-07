## TITLE: Student Motivation Level Prediction Using XGBoost  Classifier

## Objectives:
# 1.Predicting the  student motivation levels (Low, Medium, High) using academic and behavioral data.
# 2. Engineer(adding feature engineering-creating some new features for the model) relevant features to improve model performance.
# 3. Training and evaluating the model using  an XGBoost classification.
# 4. Visualize data patterns and feature importance.
# 5.Save the model for future deployment.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv(r"C:\Users\meenu\Downloads\archive (31)\Student_performance_data _.csv")
# DATASET OVERVIEW
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nColumns:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Create target
df["avg_score"] = df["GPA"] * 25
df["motivation_level"] = pd.cut(df["avg_score"], bins=[0, 50, 75, 100], labels=["Low", "Medium", "High"])
df["motivation_level"] = df["motivation_level"].astype(str)

# Feature engineering
df["study_efficiency"] = df["StudyTimeWeekly"] / (df["Absences"] + 1)
df["support_score"] = df["ParentalSupport"] + df["Tutoring"]
df["activity_score"] = df["Sports"] + df["Music"] + df["Volunteering"]

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Features
X = df[[
    "Age",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "study_efficiency",
    "support_score",
    "activity_score"
]]

y = LabelEncoder().fit_transform(df["motivation_level"])


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

# 1. Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 2. Distribution of Study Time
plt.figure()
sns.histplot(df["StudyTimeWeekly"], kde=True)
plt.title("Study Time Distribution")
plt.xlabel("StudyTimeWeekly")
plt.show()

# 3. Motivation Level Count
plt.figure()
sns.countplot(x=df["motivation_level"])
plt.title("Motivation Level Distribution")
plt.show()

# 4. Study Time vs Motivation
plt.figure()
sns.boxplot(x=df["motivation_level"], y=df["StudyTimeWeekly"])
plt.title("Study Time vs Motivation")
plt.show()

# FEATURE IMPORTANCE
importance = model.feature_importances_
features = X.columns
# sort features
indices = np.argsort(importance)
plt.figure(figsize=(10,6))
plt.barh(features[indices], importance[indices])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()