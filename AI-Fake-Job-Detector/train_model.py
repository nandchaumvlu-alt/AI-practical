import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# =====================================================
# 1. LOAD DATASET
# =====================================================
DATA_PATH = "data/fake_job_postings.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print("Shape:", df.shape)
# =====================================================
# 2. CHECK TARGET COLUMN
# =====================================================

if "fraudulent" not in df.columns:
    raise ValueError(
        "Column 'fraudulent' was not found in dataset."
    )


# =====================================================
# 3. DISPLAY CLASS DISTRIBUTION
# =====================================================

print("\nClass Distribution:")
print(df["fraudulent"].value_counts())


# =====================================================
# 4. TEXT COLUMNS
# =====================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


# Only use columns that actually exist
available_text_columns = [
    col for col in text_columns
    if col in df.columns
]

print("\nText columns being used:")
print(available_text_columns)


# =====================================================
# 5. HANDLE MISSING VALUES
# =====================================================

for col in available_text_columns:
    df[col] = df[col].fillna("").astype(str)


# =====================================================
# 6. COMBINE TEXT COLUMNS
# =====================================================

df["combined_text"] = df[
    available_text_columns
].agg(" ".join, axis=1)


# =====================================================
# 7. REMOVE EMPTY RECORDS
# =====================================================

df = df[
    df["combined_text"].str.strip() != ""
].copy()


# =====================================================
# 8. INPUT + TARGET
# =====================================================

X = df["combined_text"]

y = df["fraudulent"].astype(int)


# =====================================================
# 9. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================================
# 10. DEFINE MODELS
# =====================================================

models = {

    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),

        (
            "classifier",
            MultinomialNB()
        )
    ]),


    "Decision Tree": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),

        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=20,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]),


    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])
}


# =====================================================
# 11. TRAIN MODELS
# =====================================================

results = []

trained_models = {}

predictions = {}

probabilities = {}


for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train
    model.fit(X_train, y_train)

    trained_models[name] = model

    # Predict
    y_pred = model.predict(X_test)

    # Probability
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred

    probabilities[name] = y_prob


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# =====================================================
# 12. MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print(results_df)


# =====================================================
# 13. K-FOLD CROSS VALIDATION
# =====================================================

print("\n\n5-FOLD CROSS VALIDATION")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []


for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    print("\n", name)
    print("Fold scores:", scores)
    print("Mean F1:", scores.mean())


    cv_results.append({
        "Model": name,
        "Mean F1": scores.mean(),
        "Std F1": scores.std()
    })


cv_df = pd.DataFrame(cv_results)


# =====================================================
# 14. SAVE RESULTS
# =====================================================

import os

os.makedirs(
    "outputs",
    exist_ok=True
)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

cv_df.to_csv(
    "outputs/cross_validation.csv",
    index=False
)


# =====================================================
# 15. SAVE TEST PREDICTIONS
# =====================================================

prediction_output = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True)
})

for name, pred in predictions.items():

    prediction_output[name] = pred


prediction_output.to_csv(
    "outputs/predictions.csv",
    index=False
)
print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")