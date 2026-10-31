# ============================================================
# SPAM EMAIL DETECTION - MACHINE LEARNING PROJECT
# ============================================================
# Project Members:
# 1. Vrunda Thummar - Roll No. 525
# 2. Krisha Sojitra - Roll No. 561
#
# Dataset:
# 4270 records | 13 fields
# Target: Approved / Rejected
#
# Algorithm:
# Logistic Regression
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("                 LOAN APPROVAL PREDICTION PROJECT")
print("=" * 70)

print("Project Members:")
print("Vrunda Thummar - Roll No. 525")
print("Krisha Sojitra - Roll No. 561")
print()

print("Project: Loan Approval Prediction")
print("Dataset: Loan Approval Dataset")
print("Target Column: loan_status")
print()


# ============================================================
# 3. LOAD DATASET
# ============================================================

file_path = r"C:\\Users\\KRISHA\\Desktop\\Loan Approval Prediction [525  561]\\Dataset\\loan_approval_dataset (1).csv"
df = pd.read_csv(file_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# 4. DISPLAY FIRST RECORDS
# ============================================================

print("\n" + "=" * 50)
print("             DISPLAY FIRST 5 RECODES")
print("=" * 50)

print(df.head())


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 50)
print("             DATASET INFORMATION")
print("=" * 50)

print(df.info())


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 50)
print("             MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# ============================================================
# 7. CHECK DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 50)
print("             DUPLICATE RECODES")
print("=" * 50)

print(df.duplicated().sum())


# ============================================================
# 8. REMOVE DUPLICATE RECORDS
# ============================================================

df = df.drop_duplicates()

print("\n" + "=" * 50)
print("             REMOVE DUPLICATE RECORDS")
print("=" * 50)

print(df.shape)


# ============================================================
# 9. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 50)
print("             STATISTICAL SUMMARY")
print("=" * 50)

print(df.describe())


# ============================================================
# 10. CHECK CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 50)
print("             CATEGORICAL COLUMNS")
print("=" * 50)

print(df.select_dtypes(include="object").columns.tolist())


# ============================================================
# OUTLIER HANDLING
# ============================================================

plt.figure(figsize=(8,5))
sns.boxplot(x=df["loan_amount"])
plt.title("Before Outlier Removal")
plt.show()

Q1 = df["loan_amount"].quantile(0.25)
Q3 = df["loan_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

df = df[
    (df["loan_amount"] >= lower_bound) &
    (df["loan_amount"] <= upper_bound)
]

plt.figure(figsize=(8,5))
sns.boxplot(x=df["loan_amount"])
plt.title("After Outlier Removal")
plt.show()


# ============================================================
# 11. LOAN STATUS ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("             LOAN STATUS VALUES")
print("=" * 50)

print(df["loan_status"].value_counts())


# ============================================================
# HISTOGRAM
# ============================================================

plt.figure(figsize=(8,5))
plt.hist(df["income_amount"], bins=20)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()


# ============================================================
# 12. LOAN STATUS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="loan_status")

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.show()


# ============================================================
# 13. CONVERT CATEGORICAL VALUES
# ============================================================

df["education"] = df["education"].str.strip().str.lower()
df["self_employed"] = df["self_employed"].str.strip().str.lower()
df["loan_status"] = df["loan_status"].str.strip().str.lower()

print("\n" + "=" * 50)
print("             CATEGORICAL VALUES")
print("=" * 50)

print("\nCategorical values cleaned successfully.")


# ============================================================
# 14. ENCODE EDUCATION
# ============================================================

df["education"] = df["education"].map({
    "graduate": 1,
    "not graduate": 0
})

print("\n" + "=" * 50)
print("             EDUCATION AFTER ENCODING")
print("=" * 50)

print(df["education"].value_counts())


# ============================================================
# 15. ENCODE SELF EMPLOYED
# ============================================================

df["self_employed"] = df["self_employed"].map({
    "yes": 1,
    "no": 0
})

print("\n" + "=" * 50)
print("             SELF EMPLOYEE AFTER ENCODING")
print("=" * 50)

print(df["self_employed"].value_counts())


# ------------------------------------------------------------
# 16. PIE CHART
# ------------------------------------------------------------

class_counts = df["loan_status"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    class_counts.values,
    labels=class_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Load Approval Status Percentage")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17. LOAN AMOUNT DISTRIBUTION BY LOAN STATUS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="loan_status",
    y="loan_amount"
)

plt.title("loan Amount Distribution By Loan Status")
plt.xlabel("loan Status")
plt.ylabel("loan Amount")
plt.tight_layout()
plt.show()


# ============================================================
# 18. ENCODE LOAN STATUS
# ============================================================

df["loan_status"] = df["loan_status"].map({
    "approved": 1,
    "rejected": 0
})

print("\n" + "=" * 50)
print("             LOAN STSTUS AFTER ENCODING")
print("=" * 50)

print(df["loan_status"].value_counts())


# ============================================================
# 19. CHECK MISSING VALUES AFTER ENCODING
# ============================================================

print("\n" + "=" * 50)
print("             MISSING VALUES AFTER ENCODING")
print("=" * 50)

print(df.isnull().sum())


# ============================================================
# CORRELATION MATRIX
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()


# ============================================================
# 20. REMOVE UNNECESSARY COLUMN
# ============================================================

if "loan_id" in df.columns:
    df = df.drop("loan_id", axis=1)

print("\n" + "=" * 50)
print("             REMOVING LOAN_ID COLUMNS")
print("=" * 50)

print(df.columns.tolist())


# ============================================================
# 21. SELECT FEATURES AND TARGET
# ============================================================

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print("\n" + "=" * 50)
print("             FEATURES AND TARGET")
print("=" * 50)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("loan_status")


# ============================================================
# 22. DISPLAY X AND y
# ============================================================

print("\n" + "=" * 50)
print("             DISPLAY X AND Y")
print("=" * 50)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nX first 5 records:")
print(X.head())

print("\ny first 5 records:")
print(y.head())


# ============================================================
# 23. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("             TRAIN TEST SPLIT")
print("=" * 50)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 24. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 50)
print("             FEATURE SCALING")
print("=" * 50)

print("\nFeature scaling completed.")


# ============================================================
# 25. CREATE MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\n" + "=" * 50)
print("             CREATE ML MODEL")
print("=" * 50)

print("\nLogistic Regression model created.")


# ============================================================
# 26. TRAIN MODEL
# ============================================================

print("\n" + "=" * 50)
print("             TRAIN MODEL")
print("=" * 50)

print("\nTraining Logistic Regression model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed.")


# ============================================================
# 27. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

print("\n" + "=" * 50)
print("             MARKS PREDICTION")
print("=" * 50)

print("\nPredictions generated successfully.")

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# ============================================================
# 28. CALCULATE ACCURACY
# ============================================================

print("\n" + "=" * 50)
print("             MODEL EVALUATION")
print("=" * 50)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# ============================================================
# 29. CALCULATE PRECISION
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:")
print(precision)


# ============================================================
# 30. CALCULATE RECALL
# ============================================================

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nRecall:")
print(recall)


# ============================================================
# 31. CALCULATE F1 SCORE
# ============================================================

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nF1 Score:")
print(f1)


# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 50)
print("             CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"],
        zero_division=0
    )
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("             CONFUSION MATRIX")
print("=" * 50)

print(cm)


# ============================================================
# 34. DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()


# ============================================================
# 35. MODEL PERFORMANCE GRAPH
# ============================================================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    [value * 100 for value in metrics.values()]
)

plt.title("Loan Approval Model Performance")
plt.xlabel("Metric")
plt.ylabel("Percentage")

plt.ylim(0, 105)

for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value * 100 + 1,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ============================================================
# CROSS VALIDATION
# ============================================================

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5
)

print("\n" + "=" * 50)
print("             CROSS VALIDATION")
print("=" * 50)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage Cross Validation Accuracy:")
print(cv_scores.mean())


# ============================================================
# 36. FINAL RESULTS
# ============================================================

print("\n" + "=" * 50)
print("             FINAL MODEL RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("=" * 50)



# ============================================================
# 37. FEATURE IMPORTANCE GRAPH
# ============================================================


# Get feature importance from Logistic Regression
importance = np.abs(model.coef_[0])

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 50)
print("             FEATURE IMPORTANCE")
print("=" * 50)

print(feature_importance)

# Plot Feature Importance Graph
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Loan Approval Prediction - Feature Importance")
plt.xlabel("Absolute Logistic Regression Coefficient")
plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()



# ============================================================
# 38. PREDICT LOAN APPROVAL FOR NEW APPLICANT
# ============================================================

# 1. Check model features
print("\nDataset Features:")
print(X.columns.tolist())

# 2. Create applicant details
applicant_data = {
    "no_of_dependents": 0,
    "education": 1,
    "self_employed": 0,
    "income_annum": 5000000,
    "income_amount": 5000000,
    "loan_amount": 1500000,
    "loan_term": 12,
    "cibil_score": 750,
    "residential_assets_value": 2000000,
    "commercial_assets_value": 1000000,
    "luxury_assets_value": 3000000,
    "bank_asset_value": 1000000
}

# 3. Keep only the features required by the model
new_applicant = pd.DataFrame([{
    col: applicant_data[col]
    for col in X.columns
    if col in applicant_data
}])

# 4. Check missing features
missing_columns = [
    col for col in X.columns
    if col not in new_applicant.columns
]

if missing_columns:
    raise ValueError(
        f"Missing applicant features: {missing_columns}. "
        "Add the correct values before prediction."
    )

# 5. Match training feature order
new_applicant = new_applicant[X.columns]

# 6. Apply the trained scaler
new_applicant_scaled = scaler.transform(new_applicant)

# 7. Predict loan status
prediction = model.predict(new_applicant_scaled)[0]

# 8. Display result
print("\n" + "=" * 45)
print("       LOAN APPROVAL PREDICTION")
print("=" * 45)

if prediction == 1:
    print("Prediction: Loan Approved")
else:
    print("Prediction: Loan Rejected")

print("=" * 45)



# ------------------------------------------------------------
# 39. SAVE LOAN APPROVAL MODEL
# ------------------------------------------------------------

import pickle

with open(
    r"C:\\Users\\KRISHA\\Desktop\\Loan Approval Prediction [525  561]\\loan_approval_model.pkl",
    "wb"
) as model_file:

    pickle.dump(model, model_file)

print("Loan Approval Model saved successfully!")



# ============================================================
# 40. FINAL PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 55)
print("          LOAN APPROVAL PREDICTION PROJECT")
print("=" * 55)

print("\nProject Information:")
print("Project Name : Loan Approval Prediction")
print("Algorithm    : Logistic Regression")
print("Dataset      : Loan Approval Dataset")
print("Target       : loan_status")

print("\nModel Performance:")
print(f"Accuracy     : {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print(f"F1 Score     : {f1 * 100:.2f}%")

print("\nModel Details:")
print(f"Training Records : {len(X_train)}")
print(f"Testing Records  : {len(X_test)}")
print(f"Total Features   : {X.shape[1]}")

print("\nProject Status:")
print("Model Training      : Completed")
print("Model Evaluation    : Completed")
print("Loan Prediction     : Completed")

print("\n" + "=" * 55)
print("       PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 55)



