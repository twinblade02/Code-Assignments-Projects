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

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

#from DriftDetection import DriftAnalysis

def create_features(df, numeric_features, categorical_features):
    print("Creating robust features compatible with existing pipeline")
    
    robust_df = df.copy()
    new_numeric_features = []
    new_categorical_features = []
    
    # Ratio features, found this gem on Kaggle
    if 'MonthlyCharges' in numeric_features and 'TotalCharges' in numeric_features:
        robust_df['monthly_total_ratio'] = robust_df['MonthlyCharges'] / (robust_df['TotalCharges'] + 1)
        new_numeric_features.append('monthly_total_ratio')
        print("Added monthly_total_ratio")
    
    if 'TotalCharges' in numeric_features and 'tenure' in numeric_features:
        robust_df['charge_per_month'] = robust_df['TotalCharges'] / (robust_df['tenure'] + 1)
        new_numeric_features.append('charge_per_month')
        print("Added charge_per_month")
    
    # Service Score
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    available_services = [col for col in service_cols if col in categorical_features]
    
    if available_services:
        service_count = 0
        for col in available_services:
            service_count += (robust_df[col] == 'Yes').astype(int)
        robust_df['service_engagement'] = service_count
        new_numeric_features.append('service_engagement')
        print(f"Added service_engagement from {len(available_services)} services")
    
    # New Tenure representation
    if 'tenure' in numeric_features:
        robust_df['tenure_tier'] = pd.qcut(robust_df['tenure'], 
                                         q=4, labels=['New', 'Short', 'Medium', 'Long'], 
                                         duplicates='drop').astype(str)
        new_categorical_features.append('tenure_tier')
        print("Added tenure_tier")
    
    # Value representation
    if 'MonthlyCharges' in numeric_features:
        robust_df['value_tier'] = pd.qcut(robust_df['MonthlyCharges'], 
                                        q=3, labels=['Budget', 'Standard', 'Premium'], 
                                        duplicates='drop').astype(str)
        new_categorical_features.append('value_tier')
        print(" Added value_tier")
    
    # Customer stability score, AI suggestion
    stability_score = 0
    if 'Contract' in categorical_features:
        stability_score += (robust_df['Contract'] == 'Two year').astype(int) * 2
        stability_score += (robust_df['Contract'] == 'One year').astype(int) * 1
    
    if 'PaymentMethod' in categorical_features:
        auto_pay = robust_df['PaymentMethod'].str.contains('automatic', case=False, na=False)
        stability_score += auto_pay.astype(int)
    
    robust_df['stability_score'] = stability_score
    new_numeric_features.append('stability_score')
    print("Added stability_score")
    
    # Update feature lists
    #if not isinstance(new_numeric_features, list):
    #    new_numeric_features = []
    #if not isinstance(new_categorical_features, list):
    #    new_categorical_features = []

    updated_numeric = numeric_features + new_numeric_features
    updated_categorical = categorical_features + new_categorical_features
    
    print(f"Added {len(new_numeric_features)} numeric and {len(new_categorical_features)} categorical features")
    
    return robust_df, updated_numeric, updated_categorical

def add_adversarial_training_examples(df, augmentation_factor=0.2):
    # This is inspired by adversarial data augmentation, with some AI assistance for scenarios because I'm stupid
    print(f"Adding adversarial examples ({augmentation_factor*100:.0f}% of dataset)")
    
    base_size = len(df)
    n_augment = int(base_size * augmentation_factor)
    augmented_examples = []
    
    # Simulating a change in economy; inflation -> increase in monthly charge -> higher churn
    economic_sample = df.sample(n=n_augment//3, random_state=42).copy()
    if 'MonthlyCharges' in economic_sample.columns:
        # Customers with higher charges become more likely to churn
        high_charge_mask = economic_sample['MonthlyCharges'] > economic_sample['MonthlyCharges'].quantile(0.6)
        economic_sample.loc[high_charge_mask, 'Churn'] = 1
    augmented_examples.append(economic_sample)
    
    # Simulating new competition
    competitive_sample = df.sample(n=n_augment//3, random_state=43).copy()
    if 'InternetService' in competitive_sample.columns:
        fiber_customers = competitive_sample['InternetService'] == 'Fiber optic'
        # Simulate competitor targeting fiber customers
        competitive_sample.loc[fiber_customers, 'Churn'] = np.random.choice([0, 1], 
                                                                          size=fiber_customers.sum(), 
                                                                          p=[0.4, 0.6])
    augmented_examples.append(competitive_sample)
    
    # New services take priority -> more emphasis on Streaming -> less churn
    usage_sample = df.sample(n=n_augment//3, random_state=44).copy()
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    has_streaming = False
    for col in streaming_cols:
        if col in usage_sample.columns:
            has_streaming = usage_sample[col] == 'Yes'
            break
    
    if isinstance(has_streaming, pd.Series) and has_streaming.sum() > 0:
        usage_sample.loc[has_streaming, 'Churn'] = np.random.choice([0, 1], 
                                                                   size=has_streaming.sum(), 
                                                                   p=[0.8, 0.2])
    augmented_examples.append(usage_sample)
    
    # concat dataset
    robust_dataset = pd.concat([df] + augmented_examples, ignore_index=True)
    
    print(f"Added {len(robust_dataset) - base_size} adversarial examples")
    print(f"Final dataset size: {len(robust_dataset)} (was {base_size})")

    print(f"\nDEBUG - Feature Lists After Adv Examples:")
    print(f"robust_numeric type: {robust_dataset.dtypes}")

    return robust_dataset

def create_robust_baseline():
    print("Creating new baseline")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-robust-baseline")

    with mlflow.start_run(run_name='robust_baseline'):

    
        print("Loading data")
    
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
    
        # Repeat preprocssing pipe
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
    
        # Define feature types (same as your baseline)
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
        ]
    
        numeric_features = [f for f in numeric_features if f in df.columns and f != 'Churn']
        categorical_features = [f for f in categorical_features if f in df.columns and f != 'Churn']

        print(f'\n DEBUG - Numeric and Categoric Lists defined')

        # apply feature creation and new samples
        robust_df, robust_numeric, robust_categorical = create_features(
            df, numeric_features, categorical_features)

        print(f'\n DEBUG - Testing Dataframe integrity')
    
        robust_df = add_adversarial_training_examples(robust_df, augmentation_factor=0.15)
        print(f'\n DEBUG - Testing sample injections')
    
        # Training
        print(f'\n DEBUG - Starting training')
        X = robust_df.drop('Churn', axis=1)
        y = robust_df['Churn']

        print(f'\n DEBUG - Target dropped, XY splits created')
    
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print(f'\n DEBUG - Train-Test splits created')
    
    # Use your existing preprocessing pipeline
    #from sklearn.preprocessing import StandardScaler, OneHotEncoder
    #from sklearn.compose import ColumnTransformer
    #from sklearn.pipeline import Pipeline
    #from sklearn.ensemble import RandomForestClassifier
    
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), robust_numeric),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), robust_categorical)
        ], remainder='drop')

        print(f'\n DEBUG - Preprocessor created')
    
        robust_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            ))
        ])
    
        robust_model.fit(X_train, y_train)

        train_score = robust_model.score(X_train, y_train)
        test_score = robust_model.score(X_test, y_test)

        y_pred = robust_model.predict(X_test)
        y_prob = robust_model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_prob)

        print(f'\n DEBUG - Robust model fitted')

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("robust_features", True)

        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_auc", test_auc)

        from mlflow.models.signature import infer_signature

        mlflow.sklearn.log_model(
            robust_model, 'robustRF_model',
            signature=infer_signature(X_train, y_train), registered_model_name='telco-robust-baseline'
        )
    
        robust_training_data = X_train.copy()
        robust_training_data['Churn'] = y_train.values
        robust_training_data.to_csv("baseline_robustTrain_data.csv", index=False)
        mlflow.log_artifact('baseline_robustTrain.csv')
        print(f'\n DEBUG - Created training data artifact locally.')
    
        robust_feature_info = {
            'numeric_features': robust_numeric,
            'categorical_features': robust_categorical,
            'all_features': robust_numeric + robust_categorical,
            'n_numeric': len(robust_numeric),
            'n_categorical': len(robust_categorical),
            'n_total': len(robust_numeric) + len(robust_categorical), # this caused a damned error that took me 2 hours to debug
            'robust_features_added': True,
            'adversarial_augmentation': True,
            'augmentation_factor': 0.15
        }
    
        with open("feature_metadata_robust.json", "w") as f:
            json.dump(robust_feature_info, f, indent=2)
            print('Feature information logged.')
    
        print("Robust baseline created.")
    
    return robust_model, robust_feature_info

class RobustTrainingABTest:
    
    def __init__(self):
        self.original_analyzer = None
        self.robust_model = None
        self.results = {}
    
    def load_both_baselines(self):
        print("Loading both baseline models for comparison")
        
        # Load baseline V4
        self.original_analyzer = DriftAnalysis()
        if not self.original_analyzer.load_baseline():
            print("Cannot load original baseline")
            return False
        
        # Load robust baseline (if exists, otherwise create it)
        try:
            import json
            with open("feature_metadata_robust.json", "r") as f:
                robust_metadata = json.load(f)
            
            # Load robust model from MLflow (would be registered)
            import mlflow.sklearn
            self.robust_model = mlflow.sklearn.load_model("models:/telco-robust-baseline/latest")
            print("Loaded existing robust model from MLflow")
            
        except:
            print("Robust baseline not found, auto-creating.")
            self.robust_model, robust_metadata = create_robust_baseline()
        
        print("Baselines loaded successfully")
        return True
    
    def run_test(self):
        print("\n Running A/B test")
    
        # Generate drift for original model
        print("\n Creating drift for original model")
        original_drift_data, drift_labels = self.original_analyzer.create_drift_simulation(0.8)
    
        if original_drift_data is None:
            print("Failed to create original drift data")
            return None
    
        # Generate drift for robust model (with robust features)
        print("\n Creating drift for robust model")
        robust_drift_data, robust_drift_labels = self.original_analyzer.create_drift_simulation_with_robust_features(0.8)
        if robust_drift_data is None:
            print("Failed to create robust drift data")
            return None
    
        # Test original model
        print("\n Testing baseline model")
        try:
            original_predictions = self.original_analyzer.baseline_model.predict(original_drift_data)
            original_probs = self.original_analyzer.baseline_model.predict_proba(original_drift_data)[:,1]
            original_accuracy = accuracy_score(drift_labels, original_predictions)
            original_f1 = f1_score(drift_labels, original_predictions)
            original_auc = roc_auc_score(drift_labels, original_probs)
            print(f"Original model: Accuracy={original_accuracy:.3f}, F1={original_f1:.3f}, AUC={original_auc:.3f}")
        except Exception as e:
            print(f"Original model failed: {e}")
            return None
    
        # Test robust model
        print("\n Testing robust baseline model")
        try:
            robust_predictions = self.robust_model.predict(robust_drift_data)
            robust_probs = self.robust_model.predict_proba(robust_drift_data)[:,1]
            robust_accuracy = accuracy_score(robust_drift_labels, robust_predictions)
            robust_f1 = f1_score(robust_drift_labels, robust_predictions)
            robust_auc = roc_auc_score(robust_drift_labels, robust_probs)
            print(f"Robust model: Accuracy={robust_accuracy:.3f}, F1={robust_f1:.3f}, AUC={robust_auc:.3f}")
        except Exception as e:
            print(f"Robust model failed: {e}")
            return None
    
        # Calculate improvements
        accuracy_improvement = robust_accuracy - original_accuracy
        f1_improvement = robust_f1 - original_f1
        auc_improvement = robust_auc - original_auc
    
        print("\n Test Results:")
        print(f"Original Model: {original_accuracy:.3f} accuracy, {original_f1:.3f} F1, {original_auc:.3f} AUC")
        print(f"Robust Model:   {robust_accuracy:.3f} accuracy, {robust_f1:.3f} F1, {robust_auc:.2f} AUC")
        print(f"Improvement:    {accuracy_improvement:+.3f} accuracy, {f1_improvement:+.3f} F1, {auc_improvement:+.2f} AUC")
    
        if accuracy_improvement > 0.05:
            print("Improvement seen in Robust model")
        else:
            print("Performance is similar or worse, see results")
    
        # Return results (same format as before)
        results = {
            'original_accuracy': original_accuracy,
            'robust_accuracy': robust_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'original_f1': original_f1,
            'robust_f1': robust_f1,
            'f1_improvement': f1_improvement,
            'original_auc': original_auc,
            'robust_auc': robust_auc,
            'auc_improvement': auc_improvement
        }
    
        self.results = results
        return results
    
    def log_to_mlflow(self):
        print("\\n Logging A/B test results to MLflow")
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("telco-robust-training-ABTest")
        
        with mlflow.start_run(run_name="robust-original-Pipeline"):
            
            # Log both model performances
            mlflow.log_metric("original_accuracy", self.results['original_accuracy'])
            mlflow.log_metric("original_f1", self.results['original_f1'])
            mlflow.log_metric("original_auc", self.results['original_auc'])
            mlflow.log_metric("robust_accuracy", self.results['robust_accuracy'])
            mlflow.log_metric("robust_f1", self.results['robust_f1'])
            mlflow.log_metric("robust_auc", self.results['robust_auc'])
            
            mlflow.log_metric("accuracy_improvement", self.results['accuracy_improvement'])
            mlflow.log_metric("f1_improvement", self.results['f1_improvement'])
            mlflow.log_metric("auc_improvement", self.results['auc_improvement'])
            
            # Log experimental parameters
            mlflow.log_param("drift_strength", 0.8)
            mlflow.log_param("same_drift_thresholds", True)
            mlflow.log_param("experimental_control", "proper")
            
            # Determine winner; AI suggestion - we don't need this but okay.
            if self.results['accuracy_improvement'] > 0.05:
                winner = "robust"
            elif self.results['accuracy_improvement'] < -0.05:
                winner = "original"
            else:
                winner = "tie"
            
            mlflow.log_param("ab_test_winner", winner)
            
            print(f"A/B test results logged to MLflow")
            print(f"Winner: {winner}")

def run_robust_training_experiment():
    print("Running training experiment")
    
    # Initialize A/B test
    ab_test = RobustTrainingABTest()
    
    # Load both baselines
    if not ab_test.load_both_baselines():
        return None
    
    # Run A/B test on same drift
    results = ab_test.run_test()
    
    if results is None:
        return None
    
    # Log to MLflow
    ab_test.log_to_mlflow()
    
    # Final summary
    print("\\n Process Completed")
    
    return results

if __name__ == "__main__":
    print("Initializing test")
    
    try:
        results = run_robust_training_experiment()
        
        if results:
            print("\\n Robust training A/B test completed successfully!")
                
        else:
            print("\\n Experiment failed, please trace exceptions")
            
    except Exception as e:
        print(f"\\n Error: {e}")