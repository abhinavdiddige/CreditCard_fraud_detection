"""
Credit Card Fraud Detection - Enhanced Streamlit Application
Shows whether undersampling was used during training
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Custom CSS
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
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: black !important;
    }
    
    .info-box strong {
        color: black !important;
    }

    .info-box li {
        color: black !important;
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
            st.error(f"⚠️ Model file not found: {name}.pkl")
    
    # Load scaler
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
        st.warning("⚠️ Scaler not found")
    
    # Load label encoders
    try:
        with open('model/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    except FileNotFoundError:
        label_encoders = {}
    
    # Load feature names
    try:
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        feature_names = None
    
    return models, scaler, label_encoders, feature_names

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
    models, scaler, label_encoders, feature_names = load_models()
    comparison_df = load_comparison_results()

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📈 Model Comparison", "🔍 Fraud Prediction", "📊 Dataset Info"]
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
    
    # Training Information Box
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("⚖️ Class Imbalance Handling")
    st.write("""
    **Note**: This model was trained using one of these approaches:
    
    - **Undersampling**: Majority class reduced to match minority class (1:1 balanced)
    - **Class Weights**: All data used with balanced weights
    
    The approach used ensures fair learning of both legitimate and fraudulent transactions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset Information
    st.subheader("📦 Dataset Information")
    if feature_names:
        st.write(f"""
        **Features Used**: {len(feature_names)}
        
        **Feature List**: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}
        """)
    else:
        st.write("Dataset information will be available after training models.")

# ==================== MODEL COMPARISON PAGE ====================
elif page == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    if comparison_df is not None:
        # Info box about training approach
        st.info("📊 **Note**: These metrics reflect the chosen class balancing approach (undersampling or class weights)")
        
        st.subheader("📊 Performance Metrics Table")
        
        # Display styled dataframe
        styled_df = (
            comparison_df.style
            .highlight_max(
                axis=0,
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            )
            .set_properties(**{
                'color': 'black',              # 👈 Force text color
                'background-color': 'white'    # 👈 Force background
            })
            .format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            })
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model highlight
        best_f1_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
        best_auc_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
        best_recall_model = comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Best F1 Score", best_f1_model)
        with col2:
            st.metric("🎯 Best AUC Score", best_auc_model)
        with col3:
            st.metric("🔍 Best Recall", best_recall_model)
        with col4:
            st.metric("📊 Models Trained", len(comparison_df))
        
        # Explanation of metrics
        with st.expander("📖 Understanding the Metrics"):
            st.markdown("""
            - **Accuracy**: Overall correctness (but can be misleading with imbalanced data)
            - **AUC**: Area under ROC curve - ability to distinguish classes
            - **Precision**: Of predicted frauds, how many were actually fraud? (Fewer false alarms)
            - **Recall**: Of actual frauds, how many did we catch? (Fraud detection rate) 🎯
            - **F1 Score**: Balance between Precision and Recall
            - **MCC**: Matthews Correlation Coefficient - balanced measure for imbalanced data
            
            **For Fraud Detection**: Recall is often most important - we want to catch as many frauds as possible!
            """)
        
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
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrices
        try:
            st.subheader("🎯 Confusion Matrices")
            st.image('model/confusion_matrices.png')
            st.caption("Confusion matrices for all 6 models showing True Positives, False Positives, True Negatives, and False Negatives")
        except:
            pass
            
    else:
        st.warning("⚠️ Model comparison results not found. Please run train_models.py first.")

# ==================== FRAUD PREDICTION PAGE ====================
elif page == "🔍 Fraud Prediction":
    st.header("🔍 Fraud Detection Prediction")
    
    # Info about test data
    st.info("💡 **Tip**: Upload test data with the same features as training data. The app will handle preprocessing automatically.")
    
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
    model = models.get(selected_model_key)
    
    if model:
        st.success(f"✅ Selected Model: **{selected_model_name}**")
    else:
        st.error("❌ Model not loaded. Please run train_models.py first.")
        st.stop()
    
    # File upload
    st.subheader("📂 Upload Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing transaction data",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
            
            # Show preview
            with st.expander("👀 Preview Dataset (first 10 rows)"):
                st.dataframe(test_data.head(10))
            
            # Identify target column
            target_col = None
            possible_targets = ['is_fraud', 'isFraud', 'fraud', 'TARGET', 'Target', 'target', 'Class', 'class']
            
            for col in possible_targets:
                if col in test_data.columns:
                    target_col = col
                    break
            
            # Prepare data
            if target_col and target_col in test_data.columns:
                y_true = test_data[target_col]
                X_test = test_data.drop([target_col], axis=1)
                has_labels = True
                st.info(f"📊 Found target column: **{target_col}** - Will show performance metrics")
            else:
                X_test = test_data.copy()
                has_labels = False
                st.warning("⚠️ No target column found. Will only make predictions (no performance metrics).")
            
            # Preprocessing
            st.subheader("⚙️ Preprocessing Data")
            
            with st.spinner("Processing..."):
                # Drop ID columns
                id_cols = ['trans_num', 'cc_num', 'first', 'last', 'ID', 'Id', 'id', 'Unnamed: 0', 'index']
                drop_cols = [col for col in id_cols if col in X_test.columns]
                
                if drop_cols:
                    X_test = X_test.drop(drop_cols, axis=1)
                    st.write(f"✓ Dropped ID columns: {drop_cols}")
                
                # Encode categorical variables
                categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols and label_encoders:
                    st.write(f"✓ Encoding {len(categorical_cols)} categorical columns...")
                    for col in categorical_cols:
                        if col in label_encoders:
                            try:
                                X_test[col] = label_encoders[col].transform(X_test[col].astype(str))
                            except:
                                # Handle unseen labels
                                X_test[col] = 0
                        else:
                            # New categorical column
                            le = LabelEncoder()
                            X_test[col] = le.fit_transform(X_test[col].astype(str))
                
                # Match feature names
                if feature_names:
                    missing_cols = set(feature_names) - set(X_test.columns)
                    extra_cols = set(X_test.columns) - set(feature_names)
                    
                    if missing_cols:
                        st.warning(f"⚠️ Missing features (filled with 0): {missing_cols}")
                        for col in missing_cols:
                            X_test[col] = 0
                    
                    if extra_cols:
                        st.warning(f"⚠️ Extra features (dropped): {extra_cols}")
                        X_test = X_test.drop(columns=list(extra_cols))
                    
                    # Reorder to match training
                    X_test = X_test[feature_names]
                
                # Handle missing values
                if X_test.isnull().sum().sum() > 0:
                    st.write("✓ Filling missing values...")
                    X_test = X_test.fillna(X_test.mean())
                
                # Scale features
                if scaler:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test.values
            
            st.success("✅ Preprocessing complete!")
            
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
                    st.metric("Total Records", len(y_pred))
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
                    
                    # Interpretation
                    tn, fp, fn, tp = cm.ravel()
                    st.write(f"""
                    **Interpretation:**
                    - True Negatives (Correct Legit): {tn}
                    - False Positives (False Alarms): {fp}
                    - False Negatives (Missed Fraud): {fn} ⚠️
                    - True Positives (Caught Fraud): {tp} ✅
                    
                    **Fraud Detection Rate**: {(tp/(tp+fn)*100):.1f}% of actual frauds were caught
                    """)
                    
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
            st.error("Please ensure your CSV file has the correct format.")
            import traceback
            with st.expander("📝 See detailed error"):
                st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload a CSV file to start prediction.")

# ==================== DATASET INFO PAGE ====================
elif page == "📊 Dataset Info":
    st.header("📊 Dataset & Training Information")
    
    # Training Approach
    st.subheader("⚖️ Class Imbalance Handling")
    st.markdown("""
    <div class="info-box">
    <strong>Approach Used:</strong> This model was trained using one of these techniques:
    
    - **Undersampling**: Majority class randomly reduced to match minority class size
        - Creates perfectly balanced 1:1 dataset
        - Faster training, better fraud detection
        
    - **Class Weights**: All data used with balanced class weights
        - Keeps all information, more realistic metrics
    
    Both approaches ensure fair learning of legitimate and fraudulent transactions.
    </div>
    """, unsafe_allow_html=True)
    
    if feature_names:
        st.subheader("📋 Features Used in Model")
        st.write(f"**Total Features**: {len(feature_names)}")
        
        # Display features
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            cols[idx % 3].write(f"• {feature}")
        
        st.subheader("🔧 Preprocessing Pipeline")
        st.write("""
        1. **ID Column Removal**: Dropped transaction IDs, card numbers, personal names
        2. **Date/Time Processing**: Extracted hour, day of week, month from timestamps
        3. **Categorical Encoding**: Label encoding for categorical variables
        4. **Missing Values**: Filled with median (numerical) or mode (categorical)
        5. **Feature Scaling**: StandardScaler normalization
        6. **Class Balancing**: Undersampling or class weights applied
        """)
    else:
        st.warning("⚠️ Feature information not available. Please run train_models.py first.")
    
    if comparison_df is not None:
        st.subheader("📊 Training Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models Trained", len(comparison_df))
            st.metric("Best F1 Score", f"{comparison_df['F1'].max():.4f}")
        
        with col2:
            best_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
            st.metric("Best Overall Model", best_model)
            st.metric("Best AUC Score", f"{comparison_df['AUC'].max():.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💳 Credit Card Fraud Detection System | Built with Streamlit</p>
    <p>Machine Learning Assignment - M.Tech (AIML/DSE) | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
