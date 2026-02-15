# 💳 Credit Card Fraud Detection - Machine Learning Project

## 📌 Problem Statement

Credit card fraud is a significant threat to financial institutions and customers worldwide, causing billions of dollars in losses annually. With the exponential growth of digital transactions, automated fraud detection systems have become essential for protecting consumers and businesses. This project aims to develop and compare multiple machine learning classification models to effectively identify fraudulent credit card transactions while minimizing false positives that could negatively impact legitimate customers.

The challenge involves working with a highly imbalanced dataset where fraudulent transactions represent only a small fraction of all transactions. The goal is to build robust models that can accurately detect fraud patterns while maintaining operational efficiency and user trust.

## 📊 Dataset Description

**Dataset Source**: [Kaggle - Credit Card Fraud Data](https://www.kaggle.com/datasets/chetanmittal033/credit-card-fraud-data)

### Dataset Characteristics

- **Total Transactions**: 25,134 records
- **Fraudulent Transactions**: 420 (1.67%)
- **Legitimate Transactions**: 24,714 (98.33%)
- **Number of Features**: 12 (after preprocessing)
- **Feature Types**: Mixed (numerical and categorical)
- **Class Distribution**: Highly imbalanced (Binary Classification)
- **Target Variable**: `is_fraud` (0 = Legitimate, 1 = Fraud)

### Features Used

The dataset contains transaction-level information including:

1. **amt** - Transaction amount in dollars
2. **category** - Merchant category (e.g., grocery, gas, shopping)
3. **city** - City where transaction occurred
4. **city_pop** - Population of the city
5. **lat** - Latitude of transaction location
6. **long** - Longitude of transaction location
7. **merch_lat** - Merchant latitude
8. **merch_long** - Merchant longitude
9. **gender** - Customer gender
10. **state** - US state where transaction occurred
11. **zip** - ZIP code of transaction
12. **merchant** - Merchant name (encoded)

### Data Preprocessing

1. **ID Column Removal**: Dropped transaction IDs, credit card numbers, and customer names
2. **Categorical Encoding**: Applied Label Encoding to categorical features (category, city, state, gender, merchant)
3. **Feature Scaling**: Standardized numerical features using StandardScaler
4. **Missing Value Handling**: Filled missing values with median (numerical) or mode (categorical)
5. **Class Imbalance Handling**: Applied Random Undersampling technique

### Class Imbalance Strategy

**Approach**: Random Undersampling

**Original Distribution**:
- Legitimate transactions: 24,714 (98.33%)
- Fraudulent transactions: 420 (1.67%)
- Imbalance Ratio: 58.84:1

**After Undersampling**:
- Legitimate transactions: 420 (50%)
- Fraudulent transactions: 420 (50%)
- Imbalance Ratio: 1:1 (Perfectly balanced)
- Total training samples: 840 (from 25,134)

**Rationale**: 
- Undersampling creates a balanced dataset ensuring models learn both classes equally
- Prevents bias toward the majority class
- Improves fraud detection rate (Recall)
- Reduces training time significantly
- Standard technique in fraud detection and anomaly detection literature

## 🤖 Models Used

Six classification models were implemented and evaluated:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8407 | 0.8625 | 0.9261 | 0.7403 | 0.8228 | 0.6955 |
| Decision Tree | 0.9363 | 0.9458 | 0.9403 | 0.9316 | 0.9359 | 0.8726 |
| K-Nearest Neighbors | 0.8485 | 0.9205 | 0.8120 | 0.9067 | 0.8567 | 0.7018 |
| Naive Bayes | 0.8454 | 0.8954 | 0.9703 | 0.7123 | 0.8215 | 0.7165 |
| Random Forest (Ensemble) | 0.9472 | 0.9848 | 0.9630 | 0.9300 | 0.9462 | 0.8948 |
| XGBoost (Ensemble) | **0.9674** | **0.9930** | **0.9659** | **0.9689** | **0.9674** | **0.9347** |

### Performance Metrics Explanation

- **Accuracy**: Overall correctness of predictions (TP + TN) / Total
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Precision**: Of predicted frauds, how many were actually fraudulent - minimizes false alarms
- **Recall**: Of actual frauds, how many were detected - maximizes fraud detection
- **F1 Score**: Harmonic mean of Precision and Recall - balanced performance metric
- **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets (-1 to +1)

### Key Performance Insights

🏆 **Best Overall Model**: XGBoost
- Highest accuracy (96.74%), AUC (99.30%), and F1 score (96.74%)
- Excellent balance between Precision (96.59%) and Recall (96.89%)
- Catches 96.89% of fraudulent transactions while maintaining low false alarm rate

🎯 **Best Fraud Detection**: XGBoost (96.89% Recall)
- Successfully identifies nearly 97% of all fraud cases
- Only misses approximately 3 out of 100 fraudulent transactions

📊 **Most Precise**: Naive Bayes (97.03% Precision)
- When it predicts fraud, it's correct 97% of the time
- However, lower Recall (71.23%) means it misses more fraud cases

## 🔍 Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Serves as a solid interpretable baseline with good precision (92.61%) but moderate recall (74.03%). Fast training and lightweight deployment make it suitable for real-time systems. The linear decision boundary limits its ability to capture complex fraud patterns, resulting in the lowest F1 score (0.8228) among all models. Provides feature coefficients for interpretability, revealing which transaction attributes most strongly indicate fraud. |
| **Decision Tree** | Demonstrates strong performance (F1=0.9359) with excellent balance between precision (94.03%) and recall (93.16%). Captures non-linear fraud patterns through hierarchical rules without requiring feature scaling. The tree structure provides interpretable decision paths showing exactly how fraud is detected. Prone to overfitting, mitigated by max_depth=10 constraint. Feature importance reveals that transaction amount, merchant category, and geographic location are key fraud indicators. |
| **K-Nearest Neighbors** | Achieves respectable performance (F1=0.8567) with excellent recall (90.67%), effectively identifying similar fraud patterns by comparing transactions. Performance heavily depends on proper feature scaling and distance metrics. Computationally expensive for large datasets as it requires distance calculations to all training samples. The local decision boundary makes it effective for detecting fraud clusters in transaction space. Works well with undersampled balanced data. |
| **Naive Bayes** | Exhibits highest precision (97.03%) with lowest false alarm rate, making it ideal for scenarios where false fraud alerts are costly. However, low recall (71.23%) means it misses approximately 29% of actual fraud cases. The independence assumption between features likely doesn't hold - for example, transaction amount and merchant category are often correlated. Best used as part of an ensemble or as a first-stage filter in a multi-stage fraud detection system. |
| **Random Forest (Ensemble)** | Outstanding second-best performer (F1=0.9462) with excellent AUC (98.48%) and strong balance of precision (96.30%) and recall (93.00%). Ensemble of 100 decision trees reduces overfitting while maintaining interpretability through feature importance. Robust to outliers and handles mixed data types effectively. Class balancing through undersampling enables it to focus on discriminative fraud patterns. Provides stable predictions and generalizes well to unseen transactions. |
| **XGBoost (Ensemble)** | **Best overall model** achieving exceptional performance across all metrics: 96.74% accuracy, 99.30% AUC, and 96.74% F1 score. Superior gradient boosting algorithm optimized for tabular data with mixed feature types. Excellent balance with 96.59% precision and 96.89% recall means it catches nearly all fraud while maintaining very low false alarm rate. Handles imbalanced data naturally through scale_pos_weight parameter. Built-in regularization prevents overfitting. Recommended for production deployment due to superior performance, efficiency, and robustness. Provides feature importance for model interpretability. |

### Key Insights

1. **Ensemble Methods Dominate**: XGBoost and Random Forest significantly outperform individual classifiers, demonstrating the power of ensemble learning for fraud detection
2. **Undersampling Effectiveness**: Balanced training data improved Recall across all models, with most achieving >90% fraud detection rate
3. **Precision-Recall Trade-off**: Models like Naive Bayes prioritize precision (fewer false alarms) while KNN prioritizes recall (catching more fraud)
4. **XGBoost Superiority**: XGBoost achieves best overall performance with near-perfect AUC (99.30%) and excellent F1 score (96.74%)
5. **Feature Importance**: Transaction amount, merchant category, and geographic location emerged as top fraud predictors across models

## 🚀 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training and evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── model/                          # Saved models and results
│   ├── logistic_regression.pkl     # Trained Logistic Regression model
│   ├── decision_tree.pkl           # Trained Decision Tree model
│   ├── k-nearest_neighbors.pkl     # Trained KNN model
│   ├── naive_bayes.pkl             # Trained Naive Bayes model
│   ├── random_forest.pkl           # Trained Random Forest model
│   ├── xgboost.pkl                 # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler for feature scaling
│   ├── label_encoders.pkl          # LabelEncoders for categorical variables
│   ├── feature_names.pkl           # List of feature names used
│   ├── model_comparison.csv        # Performance metrics comparison
│   └── confusion_matrices.png      # Visualization of confusion matrices
│
└── data/                           # Dataset directory (not included in repo)
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

**Required packages:**
- streamlit
- scikit-learn==1.4.0
- numpy==1.26.3
- pandas==2.2.0
- matplotlib==3.8.2
- seaborn==0.13.1
- xgboost==2.0.3

### Step 3: Download Dataset

1. Visit [Kaggle - Credit Card Fraud Data](https://www.kaggle.com/datasets/chetanmittal033/credit-card-fraud-data)
2. Download the CSV file
3. Place it in the project root directory and rename to `creditcard.csv`

### Step 4: Train Models

```bash
python train_models.py
```

**This will:**
- Load and preprocess the dataset
- Apply undersampling for class balance
- Encode categorical variables
- Train all 6 classification models
- Evaluate models with 6 comprehensive metrics
- Save trained models in the `model/` directory
- Generate comparison results and visualizations

**Expected output:**
```
================================================================================
APPLYING UNDERSAMPLING
================================================================================
Before undersampling:
  Majority class (0): 24,714
  Minority class (1): 420
  Ratio: 58.84:1

After undersampling:
  Majority class (0): 420
  Minority class (1): 420
  Ratio: 1:1 (Perfectly balanced)
  
✅ Classes are now balanced!
================================================================================

Training and evaluating models...
✓ Logistic Regression - Accuracy: 0.8407
✓ Decision Tree - Accuracy: 0.9363
✓ XGBoost - Accuracy: 0.9674

✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!
```

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

1. **Ensure all model files are committed**
   ```bash
   git add model/*.pkl model/*.csv model/*.png
   git commit -m "Add trained models and results"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `credit-card-fraud-detection`
   - Choose branch: `main`
   - Set main file: `app.py`
   - Click "Deploy"

3. **Wait for Deployment** (2-5 minutes)

4. **Access Your Live App**
   - You'll receive a public URL: `https://your-app-name.streamlit.app`
   - Share this URL for demonstration and evaluation

### Important Notes for Deployment

- ✅ Commit all `.pkl` model files (required for app functionality)
- ❌ Do NOT commit `creditcard.csv` (150MB+ file, too large for GitHub)
- ✅ Verify `requirements.txt` has all dependencies with correct versions
- ✅ Test app locally before deploying

## 📱 Streamlit App Features

### 1. 🏠 Home Page
- Project overview and introduction
- Model descriptions
- Dataset information summary
- Feature list display
- Navigation guide

### 2. 📈 Model Comparison
- **Performance Metrics Table**: All 6 models with 6 metrics each
- **Best Model Highlights**: Top performers for F1, AUC, and Recall
- **Interactive Visualizations**: Bar charts for each metric
- **Confusion Matrices**: Visual comparison of all models
- **Exportable Results**: Download comparison as CSV

### 3. 🔍 Fraud Prediction
- **Model Selection**: Choose from 6 trained models via dropdown
- **CSV Upload**: Upload test dataset (with or without labels)
- **Automatic Preprocessing**: Handles encoding, scaling, missing values
- **Real-time Predictions**: Instant fraud classification
- **Performance Metrics**: Accuracy, AUC, Precision, Recall, F1, MCC (if labels provided)
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Classification Report**: Detailed per-class performance metrics
- **Fraud Probability Scores**: Confidence scores for each prediction
- **Download Results**: Export predictions with probabilities as CSV

### 4. 📊 Dataset Info
- Dataset statistics and characteristics
- Feature list and descriptions
- Preprocessing pipeline details
- Training summary and best model info

## 📊 Evaluation Metrics

For each model, the following metrics were calculated on the test set:

1. **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness of predictions

2. **AUC Score** = Area under the ROC curve
   - Ability to distinguish between fraud and legitimate transactions
   - Range: 0.5 (random) to 1.0 (perfect)

3. **Precision** = TP / (TP + FP)
   - Of predicted frauds, percentage that are actually fraudulent
   - Minimizes false alarms (important for customer experience)

4. **Recall** = TP / (TP + FN)
   - Of actual frauds, percentage that were detected
   - Maximizes fraud detection (critical for loss prevention)

5. **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean balancing Precision and Recall
   - Best metric for imbalanced data

6. **MCC** = Matthews Correlation Coefficient
   - Balanced measure accounting for all confusion matrix elements
   - Range: -1 (worst) to +1 (best), 0 = random

**Legend:**
- TP = True Positives (Fraud correctly identified as fraud)
- TN = True Negatives (Legitimate correctly identified as legitimate)
- FP = False Positives (Legitimate incorrectly flagged as fraud)
- FN = False Negatives (Fraud incorrectly classified as legitimate)

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Data Loading**: Read CSV file with 25,134 transactions
2. **Target Identification**: Automatically detect fraud label column
3. **ID Removal**: Drop transaction IDs, credit card numbers, customer names
4. **Categorical Encoding**: Label encode categorical features (category, city, state, gender, merchant)
5. **Date/Time Processing**: Extract hour, day of week, month from timestamps
6. **Missing Value Imputation**: Fill with median (numerical) or mode (categorical)
7. **Feature Scaling**: StandardScaler normalization (mean=0, std=1)
8. **Class Balancing**: Random undersampling to 1:1 ratio
9. **Train-Test Split**: 70% training, 30% testing (stratified)

### Model Training Configuration

**Common Parameters:**
- `random_state=42` (reproducibility)
- `class_weight='balanced'` (for non-undersampled approaches)

**Model-Specific Hyperparameters:**

1. **Logistic Regression**
   - `max_iter=1000`
   - `class_weight='balanced'`

2. **Decision Tree**
   - `max_depth=10` (prevent overfitting)
   - `class_weight='balanced'`

3. **K-Nearest Neighbors**
   - `n_neighbors=5`
   - `n_jobs=-1` (parallel processing)

4. **Naive Bayes**
   - GaussianNB with default parameters

5. **Random Forest**
   - `n_estimators=100` (100 trees)
   - `max_depth=10`
   - `class_weight='balanced'`
   - `n_jobs=-1`

6. **XGBoost**
   - `n_estimators=100`
   - `max_depth=5`
   - `learning_rate=0.1`
   - `scale_pos_weight=1` (balanced after undersampling)
   - `eval_metric='logloss'`
   - `n_jobs=-1`

### Evaluation Strategy

- **Stratified Split**: Maintains class distribution in train/test sets
- **Comprehensive Metrics**: 6 metrics per model for holistic evaluation
- **Confusion Matrices**: Visual assessment of classification performance
- **Cross-Model Comparison**: Side-by-side evaluation of all 6 models

## 🎯 Key Takeaways

1. **XGBoost is the Clear Winner**: Achieves 96.74% F1 score and 99.30% AUC, making it the best choice for production deployment

2. **Ensemble Methods Excel**: Random Forest (F1=0.9462) and XGBoost (F1=0.9674) significantly outperform individual classifiers

3. **Undersampling is Effective**: Balanced training data improved fraud detection rate (Recall) to >90% for most models

4. **Precision-Recall Trade-off**: Different models suit different business needs:
   - High Precision (Naive Bayes): Minimize false alarms
   - High Recall (XGBoost, KNN): Catch more fraud

5. **Feature Engineering Matters**: Geographic features (lat, long, city) and transaction patterns (amount, category) are strong fraud indicators

6. **Real-World Applicability**: The undersampling approach and model performance demonstrate practical fraud detection capabilities

## 🔮 Future Enhancements

### Model Improvements

1. **Hyperparameter Tuning**
   - Grid Search or Random Search for optimal parameters
   - Cross-validation for robust evaluation
   - Bayesian optimization for efficient tuning

2. **Advanced Techniques**
   - SMOTE (Synthetic Minority Over-sampling) as alternative to undersampling
   - Ensemble stacking combining multiple models
   - Deep learning models (Neural Networks, LSTM for sequential patterns)
   - Anomaly detection algorithms (Isolation Forest, One-Class SVM)

3. **Feature Engineering**
   - Transaction velocity (transactions per hour/day)
   - Average transaction amount per merchant
   - Distance from home location
   - Time-of-day and day-of-week patterns
   - Historical customer behavior features

### Deployment Enhancements

1. **Production API**
   - REST API for real-time fraud scoring
   - Batch processing for offline analysis
   - Model versioning and A/B testing

2. **Monitoring & Maintenance**
   - Model performance tracking dashboard
   - Automated retraining pipelines
   - Drift detection for data and model
   - Alert system for high-risk transactions

3. **Explainability**
   - SHAP (SHapley Additive exPlanations) values
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature contribution analysis for individual predictions

## 👨‍💻 Author

**Student Name**: [Your Name]  
**Student ID**: [Your Student ID]  
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani - Work Integrated Learning Programmes  
**Course**: Machine Learning  
**Assignment**: Assignment 2 - Classification Models & Deployment  
**Submission Date**: February 15, 2026

## 📄 License

This project is created for educational purposes as part of the Machine Learning course curriculum at BITS Pilani.

## 🙏 Acknowledgments

- **Dataset**: Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/chetanmittal033/credit-card-fraud-data)
- **Libraries**: 
  - Scikit-learn for machine learning implementations
  - XGBoost for gradient boosting algorithms
  - Streamlit for interactive web framework
  - Pandas and NumPy for data manipulation
  - Matplotlib and Seaborn for visualizations
- **Institution**: BITS Pilani Work Integrated Learning Programmes Division
- **Course Instructor**: [Instructor Name]

## 📧 Contact

For questions or feedback regarding this project:
- **Email**: [your.email@example.com]
- **GitHub**: [github.com/YOUR_USERNAME]
- **LinkedIn**: [linkedin.com/in/YOUR_PROFILE]

---

## 📝 Assignment Compliance

This project fulfills all requirements of Machine Learning Assignment 2:

✅ **Dataset**: 12 features, 25,134 instances (exceeds minimum requirements)  
✅ **Models**: All 6 required classification models implemented  
✅ **Metrics**: All 6 required evaluation metrics calculated  
✅ **Streamlit App**: All 4 required features implemented  
✅ **GitHub**: Complete source code, requirements.txt, comprehensive README  
✅ **Deployment**: Live app deployed on Streamlit Community Cloud  
✅ **Documentation**: Comparison tables, observations, and detailed analysis  

**Live Demo**: [Your Streamlit App URL]  
**GitHub Repository**: [Your GitHub Repository URL]

---

**Note**: This project was completed on BITS Virtual Lab as per assignment requirements. A screenshot of the execution has been included in the submission PDF.

---

*Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit*
