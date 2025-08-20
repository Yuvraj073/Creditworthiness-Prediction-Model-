# Loan Creditworthiness Prediction using Machine Learning

Welcome to the Loan Creditworthiness Prediction project! This repository contains code and resources for predicting loan approval decisions using advanced machine learning techniques. The primary goal is to develop a robust Random Forest model that assists financial institutions in accurate and reliable creditworthiness assessment.

## Project Overview

Loan default is a significant financial risk for lending institutions, but its impact can be substantially reduced through accurate creditworthiness prediction and data-driven decision making. This project uses Random Forest classification with hyperparameter optimization to:

- Analyze loan applicant profiles automatically
- Classify applicants as creditworthy or high-risk with high accuracy
- Provide financial institutions with reliable lending decision support
- Achieve optimal precision and recall for business applications

## Data

The dataset used for this project consists of loan applicant information organized into creditworthy and non-creditworthy cases. The data includes 18 comprehensive features covering financial, demographic, and employment information for optimal model performance.

## Model Architecture

The machine learning model consists of the following key components:

- **Random Forest Classifier**: Ensemble method with optimized hyperparameters (50-200 estimators)
- **Class Weighting**: Strategic imbalance handling (10:1 ratio) to prevent bias toward majority class
- **Preprocessing Pipeline**: StandardScaler for numeric features and OneHotEncoder for categorical variables
- **Feature Engineering**: Comprehensive feature selection covering financial and demographic aspects
- **Hyperparameter Tuning**: RandomizedSearchCV with 50 iterations for optimal parameter selection
- **Cross-Validation**: 5-fold CV strategy for robust model validation

## Key Features

- **Advanced Ensemble Architecture**: Random Forest classifier optimized for financial risk assessment
- **Class Imbalance Handling**: Weighted classification to address real-world data distribution
- **Comprehensive Preprocessing**: Automated pipeline for mixed data types and feature scaling
- **Hyperparameter Optimization**: RandomizedSearchCV with F1-score optimization for balanced performance
- **Financial Metrics**: Focus on precision and recall for lending applications
- **Probability Scoring**: Risk assessment with continuous probability outputs for decision flexibility

## Performance Metrics

The model is evaluated using financially relevant metrics:

- **Precision**: Reliability of creditworthy predictions to minimize bad loans
- **Recall (Sensitivity)**: Ability to correctly identify creditworthy applicants
- **F1-Score**: Balanced measure optimizing both precision and recall
- **Accuracy**: Overall classification performance across all cases
- **AUC-ROC Score**: Area under curve for probability-based risk assessment
- **Confusion Matrix**: Detailed breakdown of prediction accuracy by class

## How to Use

1. Clone this repository to your local machine
2. Install the required dependencies using `pip install -r requirements.txt`
3. Prepare your loan dataset as `dummy_loan_data.csv` with the specified feature structure
4. Update the file paths in the code to match your dataset location
5. Run the training script to build and train the model
6. Evaluate model performance using the provided metrics and confusion matrix analysis
7. Use the trained pipeline for inference on new loan applications with probability scores

## Requirements

```
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
numpy>=1.21.0
```

## Financial Applications

This model is designed for lending and financial risk assessment where:

- High precision minimizes approved loans that may default (false positives)
- High recall ensures creditworthy applicants are not rejected unnecessarily (false negatives)
- Automated screening can assist loan officers in high-volume application processing
- Probability scores enable flexible risk-based pricing and decision thresholds

## Contributions

Contributions to improve the project are welcome! Feel free to fork this repository, raise issues, or create pull requests.

## Disclaimer

This model is developed for research and educational purposes only. It should not be used as the sole basis for financial lending decisions. Always comply with fair lending practices and regulatory requirements when implementing automated decision systems.
