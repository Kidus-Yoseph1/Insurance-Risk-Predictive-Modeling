import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns 


print("--- Starting Task 4: Statistical Modeling ---")
try:
    # Ensure data is loaded
    df = pd.read_csv('../data/external/MachineLearningRating_v3.txt', sep='|')
    
    # Ensure TransactionMonth is a datetime object
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce') 
    
    # Ensure RegistrationYear is numeric for VehicleAge calculation
    df['RegistrationYear'] = pd.to_numeric(df['RegistrationYear'], errors='coerce') 
    
    print("Data re-loaded and critical columns converted.")
except Exception as e:
    print(f"Error loading data: {e}. Check file path.")
    
# Handle Data Quality Issues (Negative Premiums/Claims)
# For modeling, we set negative financial values to 0, assuming they are reversal errors.
df['TotalPremium'] = df['TotalPremium'].apply(lambda x: max(0, x))
df['TotalClaims'] = df['TotalClaims'].apply(lambda x: max(0, x))

# Target Variable Creation (Claim Frequency)
# Target: 1 if a claim occurred, 0 otherwise.
df['ClaimFlag'] = (df['TotalClaims'] > 0).astype(int)

# Feature Engineering (Vehicle Age)
# This line now works because TransactionMonth is confirmed datetime and RegistrationYear is numeric
df['VehicleAge'] = df['TransactionMonth'].dt.year - df['RegistrationYear']

# Feature Selection & Data Splitting
# Drop highly missing, identifier, and target leakage columns
X = df.drop(columns=[
    'TotalClaims',           # Target leakage
    'PolicyID',              # Identifier
    'TransactionMonth',      # Date feature handled by age/month
    'RegistrationYear',      # Handled by VehicleAge
    'NumberOfVehiclesInFleet', # 100% missing (from EDA)
    'ClaimFlag'              # Target variable itself
])
y = df['ClaimFlag']

# Define feature types
numerical_features = ['TotalPremium', 'SumInsured', 'CustomValueEstimate', 
                      'kilowatts', 'cubiccapacity', 'VehicleAge']
categorical_features = ['Province', 'Gender', 'MaritalStatus', 'AccountType', 'VehicleType', 'Bank']

# Handle high missing categorical features by setting missing to 'Missing' category
for col in categorical_features:
    df[col] = df[col].fillna('Missing')

# Drop numerical features that are too sparse (based on EDA findings)
numerical_features.remove('CustomValueEstimate')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dataset split: Train={len(X_train):,} rows, Test={len(X_test):,} rows.")


# PREPROCESSING PIPELINE

# Numerical Transformer: Imputation (Median) + Scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Handle low missing numerical features 
    ('scaler', StandardScaler())                    # Standardize for LR/RF
])

# Categorical Transformer: One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Column Transformer (Combines all preprocessing steps)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' 
)


# MODEL TRAINING AND EVALUATION 

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Calculates and prints performance metrics for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\n--- Evaluation: {model_name} ---")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): ({tn}, {fp}, {fn}, {tp})")


# Define Models
# Using 'balanced' class weights to account for the low frequency of claims
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    # scale_pos_weight is XGBoost's equivalent of class_weight='balanced'
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], n_jobs=-1)
}

# Iterate, Train, and Evaluate
evaluation_results = {}

for name, model in models.items():
    # Create the full pipeline: Preprocessor + Model
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])
    
    print(f"\n[Training] Starting {name}...")
    full_pipeline.fit(X_train, y_train)
    
    # Predict probabilities (for ROC AUC) and classes
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    y_pred = full_pipeline.predict(X_test)
    
    # Evaluate and store results
    evaluate_model(y_test, y_pred, y_prob, name)
    
    evaluation_results[name] = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1_score': f1_score(y_test, y_pred)
    }


# ACTIONABLE INSIGHTS (Feature Importance)

# Focus on the best model (typically XGBoost) for feature importance
best_model_name = 'XGBoost' 
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', models[best_model_name])])
xgb_pipeline.fit(X, y) # Refit on full data for stable importance

# Get feature names after one-hot encoding
feature_names = (
    numerical_features + 
    list(xgb_pipeline['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
)

# Extract importances
importance = xgb_pipeline['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False).head(15)

print("\n--- Top 15 Feature Importances (from XGBoost) ---")
print(feature_importance_df.to_markdown(numalign="left", stralign="left"))


# FINAL VISUALIZATION (Plot Feature Importance)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title(f'Top 15 Feature Importances ({best_model_name})')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
