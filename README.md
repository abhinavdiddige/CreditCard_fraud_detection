# 💳 Credit Card Fraud Detection - Machine Learning Project

## 📌 Problem Statement

Credit card fraud is a critical issue in the financial industry, causing billions of dollars in losses annually. With the increasing volume of digital transactions, automated fraud detection systems have become essential. This project aims to develop and compare multiple machine learning classification models to effectively identify fraudulent credit card transactions from legitimate ones.

The challenge lies in detecting fraudulent patterns in a highly imbalanced dataset where fraud cases represent only a tiny fraction (0.172%) of all transactions. The goal is to build models that can accurately identify fraudulent transactions while minimizing false positives and false negatives.

## 📊 Dataset Description

**Dataset Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Dataset Characteristics

- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Legitimate Transactions**: 284,315 (99.828%)
- **Number of Features**: 30
- **Feature Types**: All numerical (28 PCA components + Time + Amount)
- **Class Distribution**: Highly imbalanced (Binary Classification)
- **Time Period**: 2 days of transactions (September 2013)
- **Geographic Origin**: European cardholders

### Features

1. **V1, V2, ..., V28**: Principal components obtained with PCA transformation
   - These are anonymized features to protect confidential information
   - Result of dimensionality reduction on original features
   
2. **Time**: Seconds elapsed between each transaction and the first transaction in the dataset
   - Range: 0 to 172,792 seconds
   - Useful for temporal pattern analysis

3. **Amount**: Transaction amount
   - Range: €0.00 to €25,691.16
   - Can be used for cost-sensitive learning
   - Mean: €88.35, Median: €22.00

4. **Class**: Target variable
   - 0 = Legitimate transaction
   - 1 = Fraudulent transaction

### Dataset Challenges

- **Extreme Class Imbalance**: Only 0.172% fraud cases
- **Anonymized Features**: Limited interpretability due to PCA transformation
- **Privacy Constraints**: Original features cannot be disclosed
- **Real-world Complexity**: Actual transaction patterns from European banks

## 🤖 Models Used

The following table presents the performance comparison of all six implemented classification models:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9993 | 0.9785 | 0.8823 | 0.6394 | 0.7418 | 0.7496 |
| Decision Tree | 0.9992 | 0.9163 | 0.7824 | 0.7483 | 0.7649 | 0.7669 |
| K-Nearest Neighbors | 0.9994 | 0.9539 | 0.9195 | 0.7517 | 0.8275 | 0.8310 |
| Naive Bayes | 0.9775 | 0.9752 | 0.0622 | 0.8503 | 0.1157 | 0.2220 |
| Random Forest (Ensemble) | 0.9996 | 0.9858 | 0.9545 | 0.7824 | 0.8598 | 0.8620 |
| XGBoost (Ensemble) | 0.9996 | 0.9871 | 0.9489 | 0.8095 | 0.8736 | 0.8755 |

### Performance Metrics Explanation

- **Accuracy**: Overall correctness of predictions
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
- **Precision**: Percentage of predicted frauds that are actually fraudulent
- **Recall**: Percentage of actual frauds successfully detected
- **F1 Score**: Harmonic mean of Precision and Recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets

## 🔍 Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Provides a solid baseline with good AUC (0.9785) and interpretability. Shows moderate recall (63.94%), meaning it misses some fraud cases but maintains high precision (88.23%). Excellent for understanding linear relationships and feature importance. Training is fast and model is lightweight for deployment. |
| **Decision Tree** | Captures non-linear patterns with balanced precision (78.24%) and recall (74.83%). Good interpretability through decision rules. However, F1 score (0.7649) is lower than ensemble methods, indicating room for improvement. Prone to overfitting, which is mitigated by setting max_depth=10. Feature importance can reveal fraud detection patterns. |
| **K-Nearest Neighbors** | Strong performance with F1=0.8275 and high precision (91.95%). Effectively captures local fraud patterns by comparing similar transactions. Recall of 75.17% is decent but computationally expensive for large datasets. Performance heavily depends on proper feature scaling and choosing optimal K value (K=5 used here). |
| **Naive Bayes** | Shows excellent recall (85.03%) - best at catching actual frauds - but suffers from very low precision (6.22%), resulting in many false positives. This makes it impractical for production despite high AUC (0.9752). The independence assumption between features likely doesn't hold for fraud patterns. Best used as part of an ensemble or for initial screening. |
| **Random Forest (Ensemble)** | Outstanding performance with F1=0.8598 and excellent precision (95.45%). Reduces overfitting through ensemble of decision trees. Handles class imbalance well with class_weight='balanced'. Recall of 78.24% catches most fraud while minimizing false alarms. Top-2 performer, offering good balance between precision and recall. Robust and reliable for production deployment. |
| **XGBoost (Ensemble)** | **Best overall performer** with highest F1 score (0.8736) and MCC (0.8755). Achieves excellent balance with 94.89% precision and 80.95% recall - catches 4 out of 5 frauds while maintaining high accuracy. Superior AUC (0.9871) indicates excellent class separation. Gradient boosting with scale_pos_weight=10 handles imbalance effectively. Recommended for production deployment. |

### Key Insights

1. **Ensemble Methods Dominate**: XGBoost and Random Forest significantly outperform individual classifiers
2. **Precision-Recall Tradeoff**: Naive Bayes has high recall but poor precision; most other models balance both
3. **Class Imbalance Handling**: Models using class weighting and boosting techniques perform better
4. **Best for Production**: XGBoost offers the best overall performance for deployment
5. **Interpretability vs Performance**: Simpler models (Logistic Regression, Decision Tree) offer better interpretability but lower performance

## 🚀 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training and evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Saved models and results
│   ├── logistic_regression.pkl     # Trained Logistic Regression model
│   ├── decision_tree.pkl           # Trained Decision Tree model
│   ├── k-nearest_neighbors.pkl     # Trained KNN model
│   ├── naive_bayes.pkl             # Trained Naive Bayes model
│   ├── random_forest.pkl           # Trained Random Forest model
│   ├── xgboost.pkl                 # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler for feature scaling
│   ├── model_comparison.csv        # Performance metrics comparison
│   └── confusion_matrices.png      # Visualization of confusion matrices
│
└── data/                           # Dataset (not included in repo)
    └── creditcard.csv              # Download from Kaggle
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the project root directory

### Step 4: Train Models

```bash
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 6 classification models
- Evaluate each model with comprehensive metrics
- Save trained models in the `model/` directory
- Generate comparison results and visualizations

### Step 5: Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🌐 Deployment on Streamlit Community Cloud

### Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Set main file: `app.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - Deployment typically takes 2-5 minutes
   - Once deployed, you'll get a public URL

4. **Share Your App**
   - Copy the deployment URL
   - Share it for evaluation and demonstration

### Important Notes for Deployment

- Ensure all model files (`.pkl`) are committed to the repository
- The dataset file (`creditcard.csv`) is large (150MB) - **DO NOT** commit it to GitHub
- For the Streamlit app, only use test data uploads (smaller CSV files)
- Verify `requirements.txt` has all necessary dependencies

## 📱 Streamlit App Features

### 1. 🏠 Home Page
- Project overview and introduction
- Dataset information
- Key features summary
- Navigation guide

### 2. 📈 Model Comparison
- Performance metrics table for all 6 models
- Best model highlights
- Interactive metric visualization
- Detailed model observations

### 3. 🔍 Fraud Prediction
- **Model Selection**: Choose from 6 trained models
- **CSV Upload**: Upload test dataset (with or without labels)
- **Real-time Prediction**: Get instant fraud predictions
- **Comprehensive Metrics**: Display accuracy, AUC, precision, recall, F1, MCC
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Detailed per-class performance
- **Download Results**: Export predictions with fraud probabilities

### 4. 📊 Dataset Analysis
- Dataset statistics and overview
- Class distribution visualization
- Feature information
- Class imbalance discussion

## 📊 Evaluation Metrics

For each model, the following metrics are calculated:

1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
2. **AUC Score**: Area under the ROC curve
3. **Precision**: TP / (TP + FP) - Minimizes false alarms
4. **Recall**: TP / (TP + FN) - Maximizes fraud detection
5. **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
6. **MCC**: Balanced measure accounting for all confusion matrix elements

Where:
- TP = True Positives (Correctly identified frauds)
- TN = True Negatives (Correctly identified legitimate transactions)
- FP = False Positives (Legitimate flagged as fraud)
- FN = False Negatives (Fraud missed by model)

## 🔬 Methodology

### Data Preprocessing
1. Load creditcard.csv dataset
2. Remove 'Time' feature (not useful for fraud detection)
3. Separate features (X) and target (y)
4. Split data: 70% training, 30% testing (stratified)
5. Apply StandardScaler to normalize features

### Model Training
1. Initialize 6 classification models with optimal parameters
2. Apply class balancing techniques (class_weight='balanced')
3. Train each model on scaled training data
4. Generate predictions on test set

### Evaluation
1. Calculate 6 metrics for each model
2. Generate confusion matrices
3. Create classification reports
4. Compare model performances
5. Save results and visualizations

### Hyperparameters Used

- **Logistic Regression**: max_iter=1000, class_weight='balanced'
- **Decision Tree**: max_depth=10, class_weight='balanced'
- **KNN**: n_neighbors=5
- **Naive Bayes**: Default GaussianNB parameters
- **Random Forest**: n_estimators=100, max_depth=10, class_weight='balanced'
- **XGBoost**: n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=10

## 📝 Usage Example

```python
# Load a trained model
import pickle
with open('model/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new transaction data
new_transaction = [[...]]  # 29 features (V1-V28 + Amount)

# Scale and predict
scaled_data = scaler.transform(new_transaction)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)[:, 1]

print(f"Fraud Prediction: {prediction[0]}")
print(f"Fraud Probability: {probability[0]:.4f}")
```

## 🎯 Key Takeaways

1. **XGBoost is the Winner**: Best overall performance with F1=0.8736
2. **Ensemble Methods Excel**: Random Forest and XGBoost outperform individual classifiers
3. **Class Imbalance Matters**: Techniques like class weighting and SMOTE are crucial
4. **Metrics Beyond Accuracy**: AUC, F1, and MCC are more informative for imbalanced data
5. **Precision vs Recall**: Trade-off depends on business requirements (false alarms vs missed frauds)

## 🔮 Future Enhancements

1. **Advanced Techniques**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Cost-sensitive learning
   - Anomaly detection methods
   - Deep learning models (Neural Networks, LSTM)

2. **Feature Engineering**
   - Temporal features from Time variable
   - Transaction frequency patterns
   - Amount-based features
   - Behavioral analytics

3. **Model Optimization**
   - Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
   - Cross-validation for robust evaluation
   - Ensemble stacking
   - Model explainability (SHAP, LIME)

4. **Deployment Improvements**
   - Real-time prediction API
   - Model monitoring and retraining
   - A/B testing framework
   - Alert system for high-risk transactions

## 👨‍💻 Author

**Student Name**: [Your Name]  
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani  
**Course**: Machine Learning  
**Assignment**: Assignment 2 - Classification Models & Deployment

## 📄 License

This project is created for educational purposes as part of the Machine Learning course curriculum.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - Machine Learning Group - ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Scikit-learn library for machine learning implementations
- Streamlit for the interactive web framework
- XGBoost team for the gradient boosting library

## 📧 Contact

For questions or feedback regarding this project:
- Email: [your.email@example.com]
- GitHub: [github.com/YOUR_USERNAME]

---

**Note**: This project was completed on BITS Virtual Lab as per assignment requirements. A screenshot of the execution has been included in the submission PDF.
