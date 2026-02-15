
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """Complete pipeline for fraud detection model training and evaluation"""
    
    def __init__(self, data_path='creditcard.csv'):
        """Initialize the pipeline with dataset path"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Display dataset info
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nClass Distribution:")
        print(df['Class'].value_counts())
        print(f"\nFraud Percentage: {(df['Class'].sum() / len(df)) * 100:.2f}%")
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        # Drop 'Time' column if exists (not useful for fraud detection)
        if 'Time' in X.columns:
            X = X.drop(['Time'], axis=1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\nTraining set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize all classification models"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight='balanced'
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
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=10,
                n_jobs=-1,
                eval_metric='logloss'
            )
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate them"""
        print("\nTraining and evaluating models...\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
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
            print(f"{name} - Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            print()
            
            # Save model
            model_filename = f"model/{name.replace(' ', '_').lower()}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_filename}\n")
    
    def display_results_table(self):
        """Display comparison table of all models"""
        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        
        # Create DataFrame for better visualization
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
        
        # Save results to CSV
        results_df.to_csv('model/model_comparison.csv', index=False)
        print("\nResults saved to 'model/model_comparison.csv'")
        
        return results_df
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('model/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved to 'model/confusion_matrices.png'")
        plt.close()
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate
        self.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Display results
        results_df = self.display_results_table()
        
        # Plot confusion matrices
        self.plot_confusion_matrices()
        
        return results_df


if __name__ == "__main__":
    # Create model directory if it doesn't exist
    import os
    os.makedirs('model', exist_ok=True)
    
    print("="*100)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("="*100)
    
    # Run the pipeline
    pipeline = FraudDetectionPipeline('creditcard.csv')
    results = pipeline.run_pipeline()
    
    print("\n" + "="*100)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*100)
