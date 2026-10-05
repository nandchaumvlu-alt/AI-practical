import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🕵️",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🕵️ AI Fake Job Posting Detector")

st.markdown(
    """
    ### AI-powered job posting risk analysis

    This system uses NLP and Machine Learning to
    classify job postings as potentially fraudulent
    or likely genuine.
    """
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/fake_job_postings.csv"
    )

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

    available_columns = [
        col
        for col in text_columns
        if col in df.columns
    ]

    for col in available_columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
        )

    df["combined_text"] = (
        df[available_columns]
        .agg(" ".join, axis=1)
    )

    return df


df = load_data()


# =====================================================
# TRAIN MODELS
# =====================================================

X = df["combined_text"]

y = df["fraudulent"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {

    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),

        (
            "model",
            MultinomialNB()
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),

        (
            "model",
            DecisionTreeClassifier(
                max_depth=20,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])
}


results = {}

trained_models = {}


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    prediction = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),

        "Precision": precision_score(
            y_test,
            prediction,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            prediction,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            prediction,
            zero_division=0
        )
    }


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Analyze Job",
        "Model Performance",
        "Dataset Analysis"
    ]
)


# =====================================================
# ANALYZE JOB
# =====================================================

if page == "Analyze Job":

    st.header("🔍 Analyze a Job Posting")

    title = st.text_input(
        "Job Title"
    )

    company = st.text_input(
        "Company Profile"
    )

    location = st.text_input(
        "Location"
    )

    salary = st.text_input(
        "Salary Range"
    )

    description = st.text_area(
        "Job Description",
        height=180
    )

    requirements = st.text_area(
        "Requirements",
        height=150
    )

    benefits = st.text_area(
        "Benefits",
        height=120
    )


    if st.button(
        "🚀 Analyze Job"
    ):

        job_text = " ".join([
            title,
            company,
            location,
            salary,
            description,
            requirements,
            benefits
        ])


        if not job_text.strip():

            st.warning(
                "Please enter job details."
            )

        else:

            st.subheader(
                "🤖 AI Analysis"
            )

            predictions = {}
            probabilities = {}


            for name, model in trained_models.items():

                prediction = model.predict(
                    [job_text]
                )[0]

                probability = model.predict_proba(
                    [job_text]
                )[0][1]

                predictions[name] = prediction

                probabilities[name] = probability


            # -----------------------------------------
            # MODEL RESULTS
            # -----------------------------------------

            result_df = pd.DataFrame({

                "Model":
                list(predictions.keys()),

                "Prediction":
                [
                    "Potentially Fraudulent"
                    if x == 1
                    else "Likely Genuine"
                    for x in predictions.values()
                ],

                "Fraud Probability":
                [
                    round(x * 100, 2)
                    for x in probabilities.values()
                ]
            })


            st.dataframe(
                result_df,
                use_container_width=True
            )


            # -----------------------------------------
            # FINAL RESULT
            # -----------------------------------------

            fraud_votes = sum(
                predictions.values()
            )

            total_models = len(
                predictions
            )


            if fraud_votes > total_models / 2:

                st.error(
                    "🚨 POTENTIALLY FRAUDULENT"
                )

            elif fraud_votes == total_models / 2:

                st.warning(
                    "⚠️ MODEL DISAGREEMENT"
                )

            else:

                st.success(
                    "✅ LIKELY GENUINE"
                )


            # -----------------------------------------
            # MODEL AGREEMENT
            # -----------------------------------------

            if len(
                set(predictions.values())
            ) == 1:

                st.info(
                    "All models produced the same classification."
                )

            else:

                st.warning(
                    "Models disagree. "
                    "Additional verification is recommended."
                )


            # -----------------------------------------
            # RISK INDICATORS
            # -----------------------------------------

            st.subheader(
                "🚩 Potential Risk Indicators"
            )


            text_lower = job_text.lower()

            risk_words = {

                "registration fee":
                    "Registration/application fee mentioned",

                "pay fee":
                    "Payment requested",

                "money":
                    "Money-related language detected",

                "guaranteed":
                    "Guaranteed-income language detected",

                "earn":
                    "Earning-related claim detected",

                "no experience":
                    "No-experience claim detected",

                "work from home":
                    "Work-from-home claim detected",

                "urgent":
                    "Urgent language detected",

                "whatsapp":
                    "WhatsApp contact mentioned",

                "telegram":
                    "Telegram contact mentioned"
            }


            found_risks = []


            for keyword, message in risk_words.items():

                if keyword in text_lower:

                    found_risks.append(
                        message
                    )


            if found_risks:

                for risk in found_risks:

                    st.write(
                        "⚠️",
                        risk
                    )

            else:

                st.write(
                    "No predefined risk indicators detected."
                )


            # -----------------------------------------
            # WARNING
            # -----------------------------------------

            st.caption(
                "This system provides an AI-based risk assessment "
                "and should not be treated as definitive proof "
                "that a job is fraudulent."
            )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.header(
        "📊 Model Performance"
    )

    performance_df = pd.DataFrame(
        results
    ).T

    st.dataframe(
        performance_df.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )


    st.subheader(
        "Model Comparison"
    )

    st.bar_chart(
        performance_df[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]
    )


# =====================================================
# DATASET ANALYSIS
# =====================================================

elif page == "Dataset Analysis":

    st.header(
        "📈 Dataset Analysis"
    )


    st.subheader(
        "Class Distribution"
    )

    class_counts = (
        df["fraudulent"]
        .value_counts()
        .rename({
            0: "Genuine",
            1: "Fraudulent"
        })
    )

    st.bar_chart(
        class_counts
    )


    total = len(df)

    fraud_count = (
        df["fraudulent"] == 1
    ).sum()

    genuine_count = (
        df["fraudulent"] == 0
    ).sum()


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Genuine Jobs",
            genuine_count
        )


    with col2:

        st.metric(
            "Potentially Fraudulent Jobs",
            fraud_count
        )


    fraud_percentage = (
        fraud_count / total
    ) * 100


    if fraud_percentage < 10:

        st.warning(
            f"Fraudulent class represents only "
            f"{fraud_percentage:.2f}% of the dataset. "
            "This indicates strong class imbalance."
        )

    else:

        st.info(
            f"Fraudulent class represents "
            f"{fraud_percentage:.2f}% of the dataset."
        )