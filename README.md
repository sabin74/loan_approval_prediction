# Loan Approval Prediction

## Project Overview
This project predicts whether a loan application will be approved or not using machine learning classification models. The dataset used is from Kaggle’s Loan Prediction problem. The goal is to build a robust model to assist banks or financial institutions in making automated loan approval decisions.

---

## Tools & Libraries
- Python 3.x
- Pandas, NumPy for data processing
- Scikit-learn for model building and evaluation
- Seaborn, Matplotlib for visualization
- Joblib for model saving

---

## Steps Performed

1. **Data Loading & Exploration**  
   Explored dataset shape, features, missing values, and basic statistics.

2. **Data Cleaning & Preprocessing**  
   - Imputed missing values with median (numerical) and mode (categorical).  
   - Dropped irrelevant columns (`Loan_ID`).  
   - Encoded categorical features using Label Encoding.  
   - Scaled numerical features using StandardScaler.

3. **Exploratory Data Analysis (EDA)**  
   Visualized distributions, relationships between features and target, and detected outliers.

4. **Outlier Removal**  
   Removed outliers in `ApplicantIncome` and `LoanAmount` using the IQR method.

5. **Feature Selection**  
   Selected relevant features based on correlation and feature importance from Random Forest.

6. **Model Training**  
   Trained Logistic Regression, Random Forest, and SVM classifiers.

7. **Hyperparameter Tuning**  
   Tuned Random Forest using RandomizedSearchCV for better performance.

8. **Model Evaluation**  
   Evaluated models using accuracy, classification report, and ROC curve.

9. **Prediction on Test Data**  
   Processed test dataset similarly and predicted loan approval status.

10. **Model Saving**  
    Saved the best model using Joblib for future inference.
