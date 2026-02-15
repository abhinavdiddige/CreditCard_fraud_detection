"""
Credit Card Fraud Detection - Enhanced Streamlit Application
Theme-safe version (works in both dark & light mode)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# THEME DETECTION
# =========================
theme = st.get_option("theme.base")
is_dark = theme == "dark"

# =========================
# SAFE CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
}

.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}

.footer {
    text-align: center;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    models = {}
    model_names = [
        'logistic_regression',
        'decision_tree',
        'k-nearest_neighbors',
        'naive_bayes',
        'random_forest',
        'xgboost'
    ]

    for name in model_names:
        try:
            with open(f'model/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except:
            pass

    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    try:
        with open('model/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except:
        label_encoders = {}

    try:
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except:
        feature_names = None

    return models, scaler, label_encoders, feature_names


@st.cache_data
def load_comparison_results():
    try:
        return pd.read_csv('model/model_comparison.csv')
    except:
        return None


with st.spinner("Loading models..."):
    models, scaler, label_encoders, feature_names = load_models()
    comparison_df = load_comparison_results()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📈 Model Comparison", "🔍 Fraud Prediction", "📊 Dataset Info"]
)

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":

    st.header("Welcome")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Project Overview")
        st.write("""
        This application demonstrates 6 ML models for fraud detection:
        - Logistic Regression
        - Decision Tree
        - KNN
        - Naive Bayes
        - Random Forest
        - XGBoost
        """)

    with col2:
        st.subheader("🎯 Key Features")
        st.write("""
        - Upload custom test data
        - Model comparison
        - Confusion matrix
        - Detailed metrics
        """)

    st.info("Use the sidebar to navigate.")

# =========================
# MODEL COMPARISON
# =========================
elif page == "📈 Model Comparison":

    st.header("📈 Model Performance Comparison")

    if comparison_df is not None:

        st.info("Metrics reflect selected class balancing approach.")

        st.subheader("📊 Performance Table")

        styled_df = (
            comparison_df.style
            .highlight_max(
                axis=0,
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color="#4CAF50"
            )
            .format("{:.4f}")
        )

        # Use st.table for consistent styling across themes
        st.table(styled_df)

        # ===== Visualization =====
        st.subheader("📉 Visual Comparison")

        metric_choice = st.selectbox(
            "Select Metric:",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        )

        # Theme-safe plot style
        if is_dark:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(
            comparison_df['Model'],
            comparison_df[metric_choice]
        )

        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height:.4f}",
                ha='center',
                va='bottom'
            )

        st.pyplot(fig)

    else:
        st.warning("Model comparison file not found.")

# =========================
# FRAUD PREDICTION
# =========================
elif page == "🔍 Fraud Prediction":

    st.header("🔍 Fraud Prediction")

    selected_model_key = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    model = models.get(selected_model_key)

    if model is None:
        st.error("Model not found.")
        st.stop()

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        try:
            test_data = pd.read_csv(uploaded_file)
            st.dataframe(test_data.head())

            if st.button("Run Prediction"):

                X = test_data.copy()

                if scaler:
                    X = scaler.transform(X)

                y_pred = model.predict(X)

                st.success("Prediction completed.")

                fraud_count = np.sum(y_pred == 1)

                col1, col2 = st.columns(2)
                col1.metric("Total Records", len(y_pred))
                col2.metric("Predicted Frauds", fraud_count)

        except Exception as e:
            st.error(str(e))

# =========================
# DATASET INFO
# =========================
elif page == "📊 Dataset Info":

    st.header("Dataset & Training Info")

    st.markdown("""
    - ID columns removed
    - Label encoding applied
    - Missing values handled
    - StandardScaler normalization
    - Class imbalance handled
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<div class='footer'>💳 Built with Streamlit | BITS Pilani</div>",
    unsafe_allow_html=True
)
