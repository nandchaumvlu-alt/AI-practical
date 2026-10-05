import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>
.stApp { background: #f7f9fc; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: #111827; }
section[data-testid="stSidebar"] * { color: #f9fafb !important; }
.hero { background: linear-gradient(135deg,#111827,#334155); padding: 30px 34px; border-radius: 22px; color: white; margin-bottom: 25px; box-shadow: 0 12px 30px rgba(15,23,42,.14); }
.hero h1 { margin: 0; font-size: 40px; font-weight: 800; letter-spacing: -.8px; }
.hero p { margin: 9px 0 0; color: #dbeafe; font-size: 16px; }
.badge { display:inline-block; margin-top:15px; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.12); font-size:12px; }
.section-title { font-size: 26px; font-weight: 800; color:#111827; margin-top: 10px; }
.section-subtitle { color:#64748b; margin-bottom:20px; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:18px; padding:20px; box-shadow:0 5px 18px rgba(15,23,42,.06); height:100%; }
.label { color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.value { color:#111827; font-size:27px; font-weight:800; margin-top:5px; }
.result-fraud { background:#fff1f2; border:1px solid #fecdd3; border-left:6px solid #e11d48; border-radius:18px; padding:22px; margin:18px 0; }
.result-safe { background:#f0fdf4; border:1px solid #bbf7d0; border-left:6px solid #16a34a; border-radius:18px; padding:22px; margin:18px 0; }
.result-mixed { background:#fffbeb; border:1px solid #fde68a; border-left:6px solid #d97706; border-radius:18px; padding:22px; margin:18px 0; }
.result-title { font-size:24px; font-weight:800; }
.risk { background:#fff7ed; border:1px solid #fed7aa; border-radius:11px; padding:11px 14px; margin:7px 0; color:#9a3412; }
.safe { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:11px; padding:13px 14px; color:#166534; }
.footer { text-align:center; color:#94a3b8; font-size:12px; padding-top:30px; }
.stButton > button { border-radius:11px; font-weight:700; min-height:46px; }
div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🕵️ AI Fake Job Posting Detector</h1>
    <p>Intelligent job-posting risk analysis using NLP and Machine Learning.</p>
    <span class="badge">NLP • TF-IDF • Naive Bayes • Decision Tree • Logistic Regression</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/fake_job_postings.csv")
    text_columns = [
        "title", "company_profile", "description", "requirements", "benefits",
        "location", "department", "salary_range", "employment_type",
        "required_experience", "required_education", "industry", "function"
    ]
    available = [c for c in text_columns if c in df.columns]
    for col in available:
        df[col] = df[col].fillna("").astype(str)
    df["combined_text"] = df[available].agg(" ".join, axis=1)
    return df

df = load_data()
X = df["combined_text"]
y = df["fraudulent"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("model", MultinomialNB())
    ]),
    "Decision Tree": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("model", DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
}

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    trained = {}
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0)
        }
    return trained, results

trained_models, results = train_models(X_train, y_train, X_test, y_test)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🕵️ AI JOB GUARD")
    st.caption("Job posting risk analysis system")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Analyze Job", "Model Performance", "Dataset Analysis", "About Project"]
    )
    st.divider()
    st.markdown("### System Status")
    st.success("Dataset loaded")
    st.success("3 ML models ready")
    st.success("NLP pipeline ready")
    st.divider()
    st.caption("Academic AI/ML Project")

# =========================================================
# ANALYZE JOB
# =========================================================
if page == "Analyze Job":
    st.markdown('<div class="section-title">🔍 Analyze a Job Posting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter job details to receive an AI-based risk assessment.</div>', unsafe_allow_html=True)

    with st.form("job_form"):
        st.markdown("### 📋 Job Information")
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Job Title *", placeholder="e.g. Data Entry Operator")
            company = st.text_input("Company Profile", placeholder="Company description")
            location = st.text_input("Location", placeholder="Mumbai / Remote")
        with c2:
            salary = st.text_input("Salary Range", placeholder="e.g. ₹4-6 LPA")
            experience = st.text_input("Required Experience", placeholder="e.g. 1-3 years")
            employment = st.text_input("Employment Type", placeholder="e.g. Full-time")
        description = st.text_area("Job Description *", height=170, placeholder="Paste the job description here...")
        c3, c4 = st.columns(2)
        with c3:
            requirements = st.text_area("Requirements", height=125, placeholder="Skills, qualifications, experience...")
        with c4:
            benefits = st.text_area("Benefits", height=125, placeholder="Benefits, incentives, perks...")
        submitted = st.form_submit_button("🚀 ANALYZE JOB POSTING", use_container_width=True)

    if submitted:
        job_text = " ".join([title, company, location, salary, experience, employment, description, requirements, benefits])
        if not job_text.strip():
            st.warning("Please enter at least some job details.")
        else:
            predictions, probabilities = {}, {}
            for name, model in trained_models.items():
                predictions[name] = int(model.predict([job_text])[0])
                probabilities[name] = float(model.predict_proba([job_text])[0][1])

            fraud_votes = sum(predictions.values())
            total_models = len(predictions)
            avg_probability = np.mean(list(probabilities.values())) * 100

            if fraud_votes > total_models / 2:
                st.markdown('<div class="result-fraud"><div class="result-title">🚨 POTENTIALLY FRAUDULENT</div><div>The majority of AI models classified this posting as potentially fraudulent.</div></div>', unsafe_allow_html=True)
            elif fraud_votes == total_models / 2:
                st.markdown('<div class="result-mixed"><div class="result-title">⚠️ MODEL DISAGREEMENT</div><div>The models produced mixed predictions. Additional verification is recommended.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe"><div class="result-title">✅ LIKELY GENUINE</div><div>The majority of AI models classified this posting as likely genuine.</div></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="card"><div class="label">Average Fraud Probability</div><div class="value">{avg_probability:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="card"><div class="label">Fraud Votes</div><div class="value">{fraud_votes} / {total_models}</div></div>', unsafe_allow_html=True)
            with c3:
                agreement = "Full Agreement" if len(set(predictions.values())) == 1 else "Models Disagree"
                st.markdown(f'<div class="card"><div class="label">Model Agreement</div><div class="value" style="font-size:22px">{agreement}</div></div>', unsafe_allow_html=True)

            st.markdown("### 🧠 Model Predictions")
            cols = st.columns(3)
            for i, (name, pred) in enumerate(predictions.items()):
                with cols[i]:
                    prob = probabilities[name] * 100
                    label = "Potentially Fraudulent" if pred == 1 else "Likely Genuine"
                    icon = "🚨" if pred == 1 else "✅"
                    st.markdown(f'<div class="card"><div class="label">{name}</div><div style="font-size:18px;font-weight:750;margin-top:8px">{icon} {label}</div><div class="value">{prob:.1f}%</div><div style="color:#64748b;font-size:13px">Fraud probability</div></div>', unsafe_allow_html=True)

            st.markdown("### 🚩 Potential Risk Indicators")
            text_lower = job_text.lower()
            risk_words = {
                "registration fee": "Registration/application fee mentioned",
                "application fee": "Application fee mentioned",
                "pay fee": "Payment requested",
                "security fee": "Security fee mentioned",
                "guaranteed": "Guaranteed-income language detected",
                "earn": "Earning-related claim detected",
                "no experience": "No-experience claim detected",
                "work from home": "Work-from-home claim detected",
                "urgent": "Urgent hiring language detected",
                "whatsapp": "WhatsApp contact mentioned",
                "telegram": "Telegram contact mentioned",
                "bank details": "Bank-detail request detected",
                "aadhaar": "Identity-document request detected"
            }
            found = [msg for key, msg in risk_words.items() if key in text_lower]
            if found:
                r1, r2 = st.columns(2)
                for i, risk in enumerate(found):
                    with (r1 if i % 2 == 0 else r2):
                        st.markdown(f'<div class="risk">⚠️ {risk}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe">✓ No predefined risk indicators were detected.</div>', unsafe_allow_html=True)

            st.caption("This is an AI-based risk assessment, not definitive proof that a job is fraudulent.")

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Compare the three classification models used by the detector.</div>', unsafe_allow_html=True)

    performance_df = pd.DataFrame(results).T
    cols = st.columns(4)
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"]):
        with cols[i]:
            value = performance_df[metric].mean() * 100
            st.markdown(f'<div class="card"><div class="label">Average {metric}</div><div class="value">{value:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Detailed Comparison")
    st.dataframe((performance_df * 100).round(2), use_container_width=True)
    st.markdown("### 📈 Model Comparison")
    st.bar_chart(performance_df[["Accuracy", "Precision", "Recall", "F1 Score"]])
    st.info("Accuracy = overall correctness. Precision = correctness of fraud predictions. Recall = fraudulent jobs detected. F1 balances precision and recall.")

# =========================================================
# DATASET ANALYSIS
# =========================================================
elif page == "Dataset Analysis":
    st.markdown('<div class="section-title">📈 Dataset Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Overview of the job-posting dataset used for training.</div>', unsafe_allow_html=True)

    total = len(df)
    fraud_count = int((df["fraudulent"] == 1).sum())
    genuine_count = int((df["fraudulent"] == 0).sum())
    fraud_percentage = fraud_count / total * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card"><div class="label">Total Job Postings</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="label">Genuine Jobs</div><div class="value">{genuine_count}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="label">Fraudulent Jobs</div><div class="value">{fraud_count}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Class Distribution")
    counts = df["fraudulent"].value_counts().rename({0: "Genuine", 1: "Fraudulent"})
    st.bar_chart(counts)
    if fraud_percentage < 10:
        st.warning(f"Fraudulent postings represent {fraud_percentage:.2f}% of the dataset, indicating class imbalance.")
    else:
        st.info(f"Fraudulent postings represent {fraud_percentage:.2f}% of the dataset.")

    st.markdown("### 🗂 Dataset Preview")
    preview = [c for c in ["title", "location", "salary_range", "employment_type", "fraudulent"] if c in df.columns]
    st.dataframe(df[preview].head(20), use_container_width=True)

# =========================================================
# ABOUT
# =========================================================
elif page == "About Project":
    st.markdown('<div class="section-title">ℹ️ About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">An academic AI/ML prototype for job-posting risk analysis.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><h3>🎯 Objective</h3><p>Analyze job-posting text and classify it as potentially fraudulent or likely genuine using machine learning.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>🧠 Technologies</h3><p>Python • Pandas • Scikit-learn • TF-IDF • Naive Bayes • Decision Tree • Logistic Regression • Streamlit</p></div>', unsafe_allow_html=True)
    st.markdown("### 🔄 System Workflow")
    st.markdown('<div class="card"><b>Job Posting → Text Processing → TF-IDF → Multiple ML Models → Prediction → Fraud Probability → Risk Indicators</b></div>', unsafe_allow_html=True)
    st.warning("Use the prediction as a risk signal and verify the employer/job independently before making decisions.")

st.markdown('<div class="footer">AI Fake Job Posting Detector • Academic AI/ML Project</div>', unsafe_allow_html=True)
