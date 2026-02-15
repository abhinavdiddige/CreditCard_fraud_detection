"""
Credit Card Fraud Detection - Streamlit Web Application
Interactive dashboard for fraud detection model demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Model loading cache
@st.cache_resource
def load_models():
    """Load all trained models"""
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
        except FileNotFoundError:
            st.error(f"Model file not found: {name}.pkl")
    
    # Load scaler
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = StandardScaler()
        st.warning("Scaler not found. Using default StandardScaler.")
    
    return models, scaler

@st.cache_data
def load_comparison_results():
    """Load pre-computed model comparison results"""
    try:
        df = pd.read_csv('model/model_comparison.csv')
        return df
    except FileNotFoundError:
        return None

# Load models
with st.spinner('Loading models...'):
    models, scaler = load_models()
    comparison_df = load_comparison_results()

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📈 Model Comparison", "🔍 Fraud Prediction", "📊 Dataset Analysis"]
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome to Credit Card Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Project Overview")
        st.write("""
        This application demonstrates **6 Machine Learning classification models** 
        for detecting fraudulent credit card transactions:
        
        1. **Logistic Regression** - Linear classification baseline
        2. **Decision Tree** - Rule-based classification
        3. **K-Nearest Neighbors** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        """)
    
    with col2:
        st.subheader("🎯 Key Features")
        st.write("""
        - **Upload Custom Test Data** (CSV format)
        - **Model Selection** from 6 trained classifiers
        - **Comprehensive Metrics** (Accuracy, AUC, Precision, Recall, F1, MCC)
        - **Confusion Matrix Visualization**
        - **Classification Reports**
        - **Real-time Predictions**
        """)
    
    st.info("👈 Use the sidebar to navigate through different sections of the application.")
    
    # Dataset Information
    st.subheader("📦 Dataset Information")
    st.write("""
    **Dataset**: Credit Card Fraud Detection (Kaggle)
    
    - **Total Transactions**: 284,807
    - **Features**: 30 (V1-V28 from PCA transformation, Time, Amount)
    - **Fraudulent Transactions**: 492 (0.172%)
    - **Legitimate Transactions**: 284,315 (99.828%)
    - **Class**: Binary (0 = Normal, 1 = Fraud)
    
    This is a **highly imbalanced dataset**, making it a challenging and realistic fraud detection problem.
    """)

# ==================== MODEL COMPARISON PAGE ====================
elif page == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    if comparison_df is not None:
        st.subheader("📊 Performance Metrics Table")
        
        # Display styled dataframe
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='lightgreen')
            .format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Best model highlight
        best_f1_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
        best_auc_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best F1 Score", best_f1_model)
        with col2:
            st.metric("🎯 Best AUC Score", best_auc_model)
        with col3:
            st.metric("📊 Models Trained", len(comparison_df))
        
        # Visualization
        st.subheader("📉 Visual Comparison")
        
        metric_choice = st.selectbox(
            "Select Metric to Visualize:",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(comparison_df['Model'], comparison_df[metric_choice], color='steelblue', alpha=0.7)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_choice, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric_choice}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Model Observations
        st.subheader("🔍 Model Performance Observations")
        
        observations = {
            "Logistic Regression": "Fast and interpretable baseline model. Good for linearly separable data. May underperform on complex fraud patterns.",
            "Decision Tree": "Captures non-linear patterns but prone to overfitting. Good interpretability with feature importance.",
            "K-Nearest Neighbors": "Effective for local patterns but computationally expensive. Performance depends on proper scaling and K value.",
            "Naive Bayes": "Fast probabilistic classifier. Assumes feature independence which may not hold for fraud detection.",
            "Random Forest": "Robust ensemble method reducing overfitting. Excellent for imbalanced data with class weighting.",
            "XGBoost": "State-of-the-art gradient boosting. Handles imbalanced data well with scale_pos_weight parameter. Often achieves best performance."
        }
        
        for model, obs in observations.items():
            with st.expander(f"💡 {model}"):
                st.write(obs)
    else:
        st.warning("Model comparison results not found. Please run train_models.py first.")

# ==================== FRAUD PREDICTION PAGE ====================
elif page == "🔍 Fraud Prediction":
    st.header("🔍 Fraud Detection Prediction")
    
    # Model selection
    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'k-nearest_neighbors': 'K-Nearest Neighbors',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }
    
    selected_model_key = st.selectbox(
        "🎯 Select Model for Prediction:",
        list(model_display_names.keys()),
        format_func=lambda x: model_display_names[x]
    )
    
    selected_model_name = model_display_names[selected_model_key]
    model = models[selected_model_key]
    
    st.info(f"✅ Selected Model: **{selected_model_name}**")
    
    # File upload
    st.subheader("📂 Upload Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing test transactions",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data (excluding 'Class' column)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
            
            # Show preview
            with st.expander("👀 Preview Dataset"):
                st.dataframe(test_data.head(10))
            
            # Prepare data
            if 'Class' in test_data.columns:
                y_true = test_data['Class']
                X_test = test_data.drop(['Class'], axis=1)
                has_labels = True
            else:
                X_test = test_data.copy()
                has_labels = False
            
            # Remove Time column if exists
            if 'Time' in X_test.columns:
                X_test = X_test.drop(['Time'], axis=1)
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner('Making predictions...'):
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                st.success("✅ Prediction completed!")
                
                # Display prediction summary
                st.subheader("📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Transactions", len(y_pred))
                with col2:
                    fraud_count = np.sum(y_pred == 1)
                    st.metric("Predicted Frauds", fraud_count, delta=f"{(fraud_count/len(y_pred)*100):.2f}%")
                with col3:
                    legit_count = np.sum(y_pred == 0)
                    st.metric("Predicted Legitimate", legit_count, delta=f"{(legit_count/len(y_pred)*100):.2f}%")
                
                # If true labels available, show metrics
                if has_labels:
                    st.subheader("📈 Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        accuracy = accuracy_score(y_true, y_pred)
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        st.metric("Precision", f"{precision:.4f}")
                    
                    with col2:
                        auc = roc_auc_score(y_true, y_pred_proba)
                        st.metric("AUC Score", f"{auc:.4f}")
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        st.metric("Recall", f"{recall:.4f}")
                    
                    with col3:
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        st.metric("F1 Score", f"{f1:.4f}")
                        mcc = matthews_corrcoef(y_true, y_pred)
                        st.metric("MCC Score", f"{mcc:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("🎯 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Legitimate', 'Fraud'],
                               yticklabels=['Legitimate', 'Fraud'])
                    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
                    ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'], zero_division=0)
                    st.text(report)
                
                # Download predictions
                results_df = test_data.copy()
                results_df['Predicted_Class'] = y_pred
                results_df['Fraud_Probability'] = y_pred_proba
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"fraud_predictions_{selected_model_key}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and features.")
    
    else:
        st.info("👆 Please upload a CSV file to start prediction.")

# ==================== DATASET ANALYSIS PAGE ====================
elif page == "📊 Dataset Analysis":
    st.header("📊 Dataset Analysis")
    
    st.subheader("🔍 Dataset Overview")
    
    st.write("""
    The **Credit Card Fraud Detection Dataset** contains transactions made by credit cards in September 2013 
    by European cardholders. This dataset presents transactions that occurred over two days.
    """)
    
    # Dataset Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Dataset Statistics")
        stats_data = {
            "Metric": ["Total Transactions", "Fraudulent", "Legitimate", "Fraud Rate", "Features", "Time Period"],
            "Value": ["284,807", "492", "284,315", "0.172%", "30", "2 days"]
        }
        st.table(pd.DataFrame(stats_data))
    
    with col2:
        st.markdown("### 🎯 Class Distribution")
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [284315, 492]
        labels = ['Legitimate (99.83%)', 'Fraud (0.17%)']
        colors = ['#66b3ff', '#ff6666']
        explode = (0, 0.1)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.2f%%', shadow=True, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    st.subheader("📋 Feature Information")
    
    feature_info = pd.DataFrame({
        "Feature": ["V1-V28", "Time", "Amount", "Class"],
        "Description": [
            "Principal components from PCA transformation (anonymized features)",
            "Seconds elapsed between each transaction and first transaction",
            "Transaction amount (can be used for cost-sensitive learning)",
            "Target variable (0 = Legitimate, 1 = Fraud)"
        ],
        "Type": ["Numerical", "Numerical", "Numerical", "Categorical"]
    })
    
    st.dataframe(feature_info, use_container_width=True)
    
    st.info("""
    **Note**: Due to confidentiality, the original features have been transformed using PCA. 
    Only 'Time' and 'Amount' have not been transformed.
    """)
    
    st.subheader("⚠️ Class Imbalance Challenge")
    st.write("""
    This dataset is **highly imbalanced** with only 0.172% fraudulent transactions. 
    This presents several challenges:
    
    - Standard accuracy metric can be misleading
    - Models may bias towards the majority class
    - Requires special techniques like:
        - Class weighting
        - Resampling (SMOTE, undersampling)
        - Ensemble methods
        - Adjusted decision thresholds
        - Focus on Precision, Recall, F1, and AUC metrics
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💳 Credit Card Fraud Detection System | Built with Streamlit</p>
    <p>Machine Learning Assignment - M.Tech (AIML/DSE)</p>
</div>
""", unsafe_allow_html=True)
