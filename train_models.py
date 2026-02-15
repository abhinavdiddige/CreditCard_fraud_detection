"""
Credit Card Fraud Detection - Model Training with Undersampling
Balances classes by matching majority class to minority class size
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """Complete pipeline for fraud detection model training and evaluation"""
    
    def __init__(self, data_path='fraudTest.csv', use_undersampling=True):
        """
        Initialize the pipeline with dataset path
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file
        use_undersampling : bool
            If True, undersample majority class to match minority class
            If False, use class weights instead
        """
        self.data_path = data_path
        self.use_undersampling = use_undersampling
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def undersample_data(self, X, y):
        """
        Undersample majority class to match minority class size
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
            
        Returns:
        --------
        X_balanced, y_balanced : Balanced dataset
        """
        print("\n" + "="*80)
        print("APPLYING UNDERSAMPLING")
        print("="*80)
        
        # Combine X and y for resampling
        df = pd.concat([X, y.rename('target')], axis=1)
        
        # Separate majority and minority classes
        df_majority = df[df.target == 0]
        df_minority = df[df.target == 1]
        
        minority_count = len(df_minority)
        majority_count = len(df_majority)
        
        print(f"\nBefore undersampling:")
        print(f"  Majority class (0): {majority_count:,}")
        print(f"  Minority class (1): {minority_count:,}")
        print(f"  Ratio: {majority_count/minority_count:.2f}:1")
        print(f"  Total samples: {len(df):,}")
        
        # Undersample majority class
        df_majority_downsampled = resample(
            df_majority,
            replace=False,  # Sample without replacement
            n_samples=minority_count,  # Match minority class size
            random_state=42
        )
        
        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
        # Shuffle the dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nAfter undersampling:")
        print(f"  Majority class (0): {len(df_balanced[df_balanced.target == 0]):,}")
        print(f"  Minority class (1): {len(df_balanced[df_balanced.target == 1]):,}")
        print(f"  Ratio: 1:1 (Perfectly balanced)")
        print(f"  Total samples: {len(df_balanced):,}")
        print(f"  Reduction: {(1 - len(df_balanced)/len(df))*100:.1f}% of data removed")
        
        print("\n✅ Classes are now balanced!")
        print("="*80)
        
        # Separate features and target
        X_balanced = df_balanced.drop('target', axis=1)
        y_balanced = df_balanced['target']
        
        return X_balanced, y_balanced
        
    def load_and_preprocess_data(self):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Display dataset info
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumn Names:")
        print(list(df.columns))
        
        # Identify target column
        target_col = None
        possible_targets = ['is_fraud', 'isFraud', 'fraud', 'TARGET', 'Target', 'target', 'Class', 'class']
        
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
        
        print(f"\n✓ Target Column: '{target_col}'")
        print(f"\nClass Distribution (Original):")
        print(df[target_col].value_counts())
        fraud_pct = (df[target_col].sum() / len(df)) * 100
        print(f"\nFraud Percentage: {fraud_pct:.4f}%")
        
        # Separate features and target
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        print(f"\nInitial features: {X.shape[1]}")
        
        # Drop ID columns
        drop_cols = []
        strict_id_cols = ['trans_num', 'cc_num', 'first', 'last']  # Keep merchant and job
        
        for col in strict_id_cols:
            if col in X.columns:
                drop_cols.append(col)
        
        for col in X.columns:
            if 'Unnamed' in col or col in ['index', 'ID', 'Id', 'id']:
                drop_cols.append(col)
        
        if drop_cols:
            X = X.drop(drop_cols, axis=1)
            print(f"Dropped ID columns: {drop_cols}")
        
        print(f"\nFeatures after dropping IDs: {X.shape[1]}")
        
        # Handle date/time columns
        categorical_cols = []
        numerical_cols = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        X[col] = pd.to_datetime(X[col])
                        X[f'{col}_hour'] = X[col].dt.hour
                        X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                        X[f'{col}_month'] = X[col].dt.month
                        X = X.drop([col], axis=1)
                        numerical_cols.extend([f'{col}_hour', f'{col}_dayofweek', f'{col}_month'])
                        print(f"Extracted time features from: {col}")
                    except:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
            elif X[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
        
        print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        
        # Encode categorical variables
        if categorical_cols:
            print("\nEncoding categorical variables...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col}: {len(le.classes_)} unique values")
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("\nHandling missing values...")
            for col in numerical_cols:
                if col in X.columns and X[col].isnull().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            for col in categorical_cols:
                if col in X.columns and X[col].isnull().sum() > 0:
                    X[col] = X[col].fillna(X[col].mode()[0])
            print(f"  ✓ Missing values handled")
        
        print(f"\n{'='*80}")
        print(f"FINAL FEATURE COUNT: {X.shape[1]} features")
        print(f"{'='*80}")
        
        if X.shape[1] < 12:
            raise ValueError(f"Insufficient features: {X.shape[1]} < 12 required")
        else:
            print(f"\n✅ Feature requirement met: {X.shape[1]} >= 12")
        
        print(f"\nFeatures used: {list(X.columns)}")
        
        # APPLY UNDERSAMPLING if enabled
        if self.use_undersampling:
            X, y = self.undersample_data(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save feature names
        self.feature_names = list(X.columns)
        
        # Save scaler and encoders
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('model/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open('model/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"\nTraining set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"\nTraining set distribution:")
        print(f"  Class 0: {(y_train == 0).sum():,}")
        print(f"  Class 1: {(y_train == 1).sum():,}")
        
        # Calculate scale_pos_weight for XGBoost (only if not undersampling)
        if self.use_undersampling:
            scale_pos_weight = 1  # Balanced data
        else:
            fraud_rate = fraud_pct / 100
            scale_pos_weight = int((1 - fraud_rate) / fraud_rate) if fraud_rate > 0 else 1
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scale_pos_weight
    
    def initialize_models(self, scale_pos_weight=1):
        """Initialize all classification models"""
        print("\nInitializing models...")
        
        # If undersampling, don't use class_weight='balanced'
        if self.use_undersampling:
            class_weight = None
            print("Using balanced data (no class weights needed)")
        else:
            class_weight = 'balanced'
            print("Using class weights for imbalanced data")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight=class_weight
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight=class_weight
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                eval_metric='logloss'
            )
        }
        
        print(f"✓ XGBoost scale_pos_weight: {scale_pos_weight}")
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate them"""
        print("\nTraining and evaluating models...\n")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC': roc_auc_score(y_test, y_pred_proba),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'MCC': matthews_corrcoef(y_test, y_pred)
            }
            
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, zero_division=0)
            }
            
            # Display metrics
            print(f"✓ {name} - Metrics:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name:12s}: {value:.4f}")
            
            # Save model
            model_filename = f"model/{name.replace(' ', '_').lower()}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"    Saved to: {model_filename}")
        
        print("\n" + "="*80)
    
    def display_results_table(self):
        """Display comparison table of all models"""
        print("\n" + "="*100)
        print(" "*35 + "MODEL COMPARISON TABLE")
        print("="*100)
        
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['metrics']['Accuracy'] for m in self.results],
            'AUC': [self.results[m]['metrics']['AUC'] for m in self.results],
            'Precision': [self.results[m]['metrics']['Precision'] for m in self.results],
            'Recall': [self.results[m]['metrics']['Recall'] for m in self.results],
            'F1': [self.results[m]['metrics']['F1'] for m in self.results],
            'MCC': [self.results[m]['metrics']['MCC'] for m in self.results]
        })
        
        print(results_df.to_string(index=False))
        print("="*100)
        
        results_df.to_csv('model/model_comparison.csv', index=False)
        print("\n✓ Results saved to 'model/model_comparison.csv'")
        
        return results_df
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=True)
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_xticklabels(['Legitimate', 'Fraud'])
            axes[idx].set_yticklabels(['Legitimate', 'Fraud'])
        
        plt.tight_layout()
        plt.savefig('model/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved to 'model/confusion_matrices.png'")
        plt.close()
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        X_train, X_test, y_train, y_test, scale_pos_weight = self.load_and_preprocess_data()
        self.initialize_models(scale_pos_weight)
        self.train_and_evaluate(X_train, X_test, y_train, y_test)
        results_df = self.display_results_table()
        self.plot_confusion_matrices()
        return results_df


if __name__ == "__main__":
    import os
    os.makedirs('model', exist_ok=True)
    
    print("\n" + "="*100)
    print(" "*25 + "CREDIT CARD FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("="*100)
    
    # CONFIGURATION: Set to True to use undersampling, False to use class weights
    USE_UNDERSAMPLING = True  # ← CHANGE THIS TO False to disable undersampling
    
    if USE_UNDERSAMPLING:
        print("\n⚖️  UNDERSAMPLING ENABLED: Majority class will be reduced to match minority class")
    else:
        print("\n⚙️  CLASS WEIGHTS ENABLED: Using balanced class weights instead of undersampling")
    
    try:
        pipeline = FraudDetectionPipeline('fraudTest.csv', use_undersampling=USE_UNDERSAMPLING)
        results = pipeline.run_pipeline()
        
        print("\n" + "="*100)
        print(" "*30 + "✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*100)
        print("\nGenerated files in model/ directory:")
        print("  • 6 trained model files (.pkl)")
        print("  • scaler.pkl")
        print("  • label_encoders.pkl")
        print("  • feature_names.pkl")
        print("  • model_comparison.csv")
        print("  • confusion_matrices.png")
        
        if USE_UNDERSAMPLING:
            print("\n📊 Note: Models were trained on BALANCED data (undersampling applied)")
        else:
            print("\n📊 Note: Models were trained on IMBALANCED data (class weights applied)")
        
        print("\nNext step: Run 'streamlit run app.py' to test the web application!")
        print("="*100 + "\n")
    
    except Exception as e:
        print("\n" + "="*100)
        print(" "*35 + "❌ ERROR OCCURRED")
        print("="*100)
        print(f"\n{str(e)}")
        print("\n" + "="*100 + "\n")
