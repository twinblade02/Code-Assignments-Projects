import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import json
from pathlib import Path
from dotenv import load_dotenv
import os

def create_robust_features(df, numeric_features, categorical_features):
    """Create robust features for better drift resistance"""
    print("Creating robust features...")
    
    robust_df = df.copy()
    new_numeric = []
    new_categorical = []
    
    # Ratio features (less sensitive to scaling drift)
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        robust_df['monthly_total_ratio'] = robust_df['MonthlyCharges'] / (robust_df['TotalCharges'] + 1)
        new_numeric.append('monthly_total_ratio')
        print("Added monthly_total_ratio")
    
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        robust_df['charge_per_month'] = robust_df['TotalCharges'] / (robust_df['tenure'] + 1)
        new_numeric.append('charge_per_month')
        print("Added charge_per_month")
    
    # Service engagement score (aggregated feature)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    available_services = [col for col in service_cols if col in df.columns]
    
    if available_services:
        service_count = sum((robust_df[col] == 'Yes').astype(int) for col in available_services)
        robust_df['service_engagement'] = service_count
        new_numeric.append('service_engagement')
        print(f"Added service_engagement from {len(available_services)} services")
    
    # Binned features (less sensitive to outliers)
    if 'tenure' in df.columns:
        robust_df['tenure_tier'] = pd.qcut(robust_df['tenure'], 
                                         q=4, labels=['New', 'Short', 'Medium', 'Long'], 
                                         duplicates='drop').astype(str)
        new_categorical.append('tenure_tier')
        print("Added tenure_tier")
    
    if 'MonthlyCharges' in df.columns:
        robust_df['value_tier'] = pd.qcut(robust_df['MonthlyCharges'], 
                                        q=3, labels=['Budget', 'Standard', 'Premium'], 
                                        duplicates='drop').astype(str)
        new_categorical.append('value_tier')
        print("Added value_tier")
    
    # Composite stability score
    stability_score = np.zeros(len(robust_df))
    if 'Contract' in df.columns:
        stability_score += (robust_df['Contract'] == 'Two year').astype(int) * 2
        stability_score += (robust_df['Contract'] == 'One year').astype(int) * 1
    
    if 'PaymentMethod' in df.columns:
        auto_pay = robust_df['PaymentMethod'].str.contains('automatic', case=False, na=False)
        stability_score += auto_pay.astype(int)
    
    robust_df['stability_score'] = stability_score
    new_numeric.append('stability_score')
    print("Added stability_score")
    
    updated_numeric = numeric_features + new_numeric
    updated_categorical = categorical_features + new_categorical
    
    print(f"Added {len(new_numeric)} numeric and {len(new_categorical)} categorical features")
    return robust_df, updated_numeric, updated_categorical

def add_adversarial_examples(df, augmentation_factor=0.15):
    """Add adversarial training examples for robustness"""
    print(f"Adding adversarial examples ({augmentation_factor*100:.0f}% of dataset)")
    
    base_size = len(df)
    n_augment = int(base_size * augmentation_factor)
    augmented_examples = []
    
    # Economic stress scenario
    economic_sample = df.sample(n=n_augment//3, random_state=42).copy()
    if 'MonthlyCharges' in economic_sample.columns:
        high_charge_mask = economic_sample['MonthlyCharges'] > economic_sample['MonthlyCharges'].quantile(0.6)
        economic_sample.loc[high_charge_mask, 'Churn'] = 1
    augmented_examples.append(economic_sample)
    
    # Competition scenario
    competitive_sample = df.sample(n=n_augment//3, random_state=43).copy()
    if 'InternetService' in competitive_sample.columns:
        fiber_customers = competitive_sample['InternetService'] == 'Fiber optic'
        competitive_sample.loc[fiber_customers, 'Churn'] = np.random.choice([0, 1], 
                                                                          size=fiber_customers.sum(), 
                                                                          p=[0.4, 0.6])
    augmented_examples.append(competitive_sample)
    
    # Service engagement scenario
    usage_sample = df.sample(n=n_augment//3, random_state=44).copy()
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    for col in streaming_cols:
        if col in usage_sample.columns:
            has_streaming = usage_sample[col] == 'Yes'
            usage_sample.loc[has_streaming, 'Churn'] = np.random.choice([0, 1], 
                                                                       size=has_streaming.sum(), 
                                                                       p=[0.8, 0.2])
            break
    augmented_examples.append(usage_sample)
    
    robust_dataset = pd.concat([df] + augmented_examples, ignore_index=True)
    print(f"Dataset size: {len(df)} → {len(robust_dataset)}")
    
    return robust_dataset

def train_robust_model(file_path="C:/Users/ldmag/Documents/GitHub/Code-Assignments-Projects/Projects/MLOps Drift Detection and Pipeline Optimization/data/Telco-Churn.csv"):
    """Train robust baseline model"""
    print("=== Training Robust Baseline Model ===")
    
    # Setup MLflow
    load_dotenv()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-baseline") # use the same experiment so we can log model runs
    
    with mlflow.start_run(run_name='robust_baseline_model'):
        # Load and preprocess data
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Data preprocessing
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Define feature types
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
        ]
        
        # Filter existing columns
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Create robust features
        robust_df, robust_numeric, robust_categorical = create_robust_features(
            df, numeric_features, categorical_features)
        
        # Add adversarial examples
        robust_df = add_adversarial_examples(robust_df, augmentation_factor=0.15)
        
        # Prepare training data
        X = robust_df.drop('Churn', axis=1)
        y = robust_df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create robust pipeline
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), robust_numeric),  # RobustScaler is less sensitive to outliers
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             robust_categorical)
        ], remainder='drop')
        
        robust_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150,  # More trees for stability
                max_depth=12,      # Slightly deeper
                min_samples_split=5,  # More conservative splitting
                min_samples_leaf=3,   # Prevent overfitting
                random_state=42, 
                class_weight='balanced'
            ))
        ])
        
        # Train model
        print("Training robust model...")
        robust_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = robust_model.score(X_train, y_train)
        test_score = robust_model.score(X_test, y_test)
        
        y_pred = robust_model.predict(X_test)
        y_prob = robust_model.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Train Score: {train_score:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test F1: {test_f1:.3f}")  
        print(f"Test AUC: {test_auc:.3f}")
        
        # Log metrics
        mlflow.log_param("model_type", "RandomForest")
        #mlflow.log_param("robust_features", True)
        #mlflow.log_param("adversarial_augmentation", True)
        mlflow.log_param("n_estimators", 150)
        #mlflow.log_param("scaler_type", "RobustScaler")
        
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        mlflow.log_metric("churn_rate", df['Churn'].mean())
        
        # Save model
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_train)
        
        mlflow.sklearn.log_model(
            robust_model, 
            'robust_model',
            signature=signature, 
            registered_model_name='telco-robust-baseline'
        )
        
        # Save training data and metadata
        robust_training_data = X_train.copy()
        robust_training_data['Churn'] = y_train.values
        robust_training_data.to_csv("robust_training_data.csv", index=False)
        mlflow.log_artifact("robust_training_data.csv")
        
        # Save feature metadata
        robust_feature_info = {
            'numeric_features': robust_numeric,
            'categorical_features': robust_categorical,
            'all_features': robust_numeric + robust_categorical,
            'n_numeric_features': len(robust_numeric),
            'n_categorical_features': len(robust_categorical),
            'n_total': len(robust_numeric) + len(robust_categorical),
            'robust_features_added': True,
            'adversarial_augmentation': True,
            'augmentation_factor': 0.15
        }
        
        with open("feature_metadata_robust.json", "w") as f:
            json.dump(robust_feature_info, f, indent=2)
        
        mlflow.log_artifact("feature_metadata_robust.json")
        
        print("Robust baseline model trained and logged successfully!")
        return robust_model, robust_feature_info

if __name__ == "__main__":
    print("Starting robust baseline training...")
    try:
        model, metadata = train_robust_model()
        print("Robust model training completed!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()