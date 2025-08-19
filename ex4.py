import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint

# Load the dataset
data = pd.read_csv('dummy_loan_data.csv')

# Define feature columns and target
feature_cols = [
    'Age', 'Employment_Status', 'Education_Level', 'Credit_Score', 
    'Credit_History_Length', 'Num_Current_Loans', 'Loan_Default_History',
    'Annual_Income', 'Debt_to_Income_Ratio', 'Loan_Amount_Requested', 
    'Monthly_Debt_Obligations', 'Job_Tenure', 'Industry', 
    'Home_Ownership_Status', 'Value_of_Owned_Property', 'Savings_Investments',
    'Marital_Status', 'Num_Dependents'
]
target_col = 'Creditworthy'

features = data[feature_cols]
target = data[target_col]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = [
    'Age', 'Credit_Score', 'Credit_History_Length', 'Num_Current_Loans',
    'Annual_Income', 'Debt_to_Income_Ratio', 'Loan_Amount_Requested',
    'Monthly_Debt_Obligations', 'Job_Tenure', 'Value_of_Owned_Property',
    'Savings_Investments', 'Num_Dependents'
]
categorical_features = [
    'Employment_Status', 'Education_Level', 'Industry', 
    'Home_Ownership_Status', 'Marital_Status'
]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the RandomForest model with class weights
rf_model = RandomForestClassifier(random_state=42, class_weight={0: 10, 1: 1})

# Define the hyperparameter grid
param_distributions = {
    'model__n_estimators': randint(50, 201),  # Random integer between 50 and 200
    'model__max_depth': [None] + list(range(10, 31)),  # None or integers between 10 and 30
    'model__min_samples_split': randint(2, 11)  # Random integer between 2 and 10
}

# Combine preprocessing and modeling into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# Initialize and fit RandomizedSearchCV with class weights
random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, scoring='f1', n_jobs=1, random_state=42)
random_search.fit(X_train, y_train)

# Predict on the test set with the best model
y_pred = random_search.best_estimator_.predict(X_test)
y_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]  # Probability scores for the positive class

# Evaluate the model
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nAUC-ROC Score:", roc_auc_score(y_test, y_proba))

# Display a few predictions with probability scores
predictions_with_probs = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_proba
})
print("\nSample Predictions with Probability Scores:\n", predictions_with_probs.head())
