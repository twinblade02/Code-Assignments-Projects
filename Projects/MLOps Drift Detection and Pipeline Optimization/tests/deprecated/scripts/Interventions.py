import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time
import json
import os
import boto3
from dotenv import load_dotenv
from pathlib import Path

from DriftDetection import DriftAnalysis

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

class DriftInterventions:  
    def __init__(self):
        self.baseline_model = None
        self.baseline_data = None
        self.drift_data = None
        self.drift_labels = None
        self.intervention_results = {}
        
    def load_baseline_and_drift(self):
        print("Loading baseline model and generating drift data")
        
        drift_analyzer = DriftAnalysis()
        
        if not drift_analyzer.load_baseline():
            print("Cannot load baseline. Run baseline_fixed.py first!")
            return False
            
        # Generate drift data
        self.drift_data, self.drift_labels = drift_analyzer.create_drift_simulation(0.5) # need to change this value each time we want to change strength
        
        if self.drift_data is None:
            print("Failed to generate drift data!")
            return False
            
        # Store references
        self.baseline_data = drift_analyzer.baseline_data
        self.baseline_model = drift_analyzer.baseline_model
        
        print(f"Baseline data: {self.baseline_data.shape}")
        print(f"Drift data: {self.drift_data.shape}")
        print(f"Baseline model loaded")
        
        return True
    
    def test_intervention_retraining(self):
        # test retraining strategy
        print("\\n Testing RETRAINING intervention")
        
        start_time = time.time()
        
        # split
        X_drift_train, X_drift_test, y_drift_train, y_drift_test = train_test_split(
            self.drift_data, self.drift_labels, 
            test_size=0.3, random_state=42, stratify=self.drift_labels
        )
        
        # retrain
        print("Retraining model on drift data")
        retrained_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        
        if hasattr(self.baseline_model, 'named_steps'):
            # Use the same preprocessing pipeline
            from sklearn.pipeline import Pipeline
            retrained_pipeline = Pipeline([
                ('preprocessor', self.baseline_model.named_steps['preprocessor']),
                ('classifier', retrained_model)
            ])
            
            retrained_pipeline.fit(X_drift_train, y_drift_train)
            retrained_predictions = retrained_pipeline.predict(X_drift_test)
            y_pred_proba = retrained_pipeline.predict_proba(X_drift_test)[:, 1]
            
            model_to_save = retrained_pipeline
            print('Pipeline retrained')
            
        else:
            # Direct model training
            retrained_model.fit(X_drift_train, y_drift_train)
            retrained_predictions = retrained_model.predict(X_drift_test)
            y_pred_proba = retrained_model.predict_proba(X_drift_test)[:, 1]
            model_to_save = retrained_model
            print('Model was retrained directly from artifact, sklearn pipeline was not used')
        
        training_time = time.time() - start_time
        
        # Evaluate retrained model
        retrained_accuracy = accuracy_score(y_drift_test, retrained_predictions)
        retrained_auc = roc_auc_score(y_drift_test, y_pred_proba)
        retrained_f1 = f1_score(y_drift_test, retrained_predictions)
        
        # Compare with baseline performance on same drift data
        baseline_predictions = self.baseline_model.predict(X_drift_test)
        baseline_accuracy = accuracy_score(y_drift_test, baseline_predictions)
        baseline_f1 = f1_score(y_drift_test, baseline_predictions)
        baseline_auc = roc_auc_score(y_drift_test, y_pred_proba)
        
        accuracy_improvement = retrained_accuracy - baseline_accuracy
        f1_improvement = retrained_f1 - baseline_f1
        auc_improvement = retrained_auc - baseline_auc
        
        results = {
            'intervention_type': 'retraining',
            'retrained_accuracy': retrained_accuracy,
            'retrained_f1': retrained_f1,
            'retrained_auc': retrained_auc,
            'baseline_accuracy_on_drift': baseline_accuracy,
            'baseline_f1_on_drift': baseline_f1,
            'baseline_auc_on_drift': baseline_auc, 
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement,
            'auc_improvement': auc_improvement,
            'training_time_seconds': training_time,
            'training_samples': len(X_drift_train)
        }
        
        print(f"Retraining complete! Time: {training_time:.1f}s")
        print(f"Baseline accuracy on drift: {baseline_accuracy:.3f}")
        print(f"Retrained accuracy: {retrained_accuracy:.3f}")
        print(f"Improvement: {accuracy_improvement:+.3f} ({accuracy_improvement/baseline_accuracy*100:+.1f}%)")
        
        self.intervention_results['retraining'] = results
        return model_to_save, results
    
    def test_intervention_rollback(self):
        print("\\n Testing ROLLBACK intervention")
        
        # Since we have just one baseline with a single version, we will need to simulate a rollback by testing it on drifted data instead.
        # Another limitation - baseline used the entire dataset, so we have no additional data that is unseen. Could actually do this for a future version.
        
        rollback_predictions = self.baseline_model.predict(self.drift_data)
        y_pred_proba = self.baseline_model.predict_proba(self.drift_data)[:, 1]

        rollback_accuracy = accuracy_score(self.drift_labels, rollback_predictions)
        rollback_auc = roc_auc_score(self.drift_labels, y_pred_proba)
        rollback_f1 = f1_score(self.drift_labels, rollback_predictions)
        
        # Calculate degradation from original baseline performance
        original_baseline_accuracy = 0.76
        accuracy_degradation = original_baseline_accuracy - rollback_accuracy #this should in fact be the same
        
        results = {
            'intervention_type': 'rollback',
            'rollback_accuracy': rollback_accuracy,
            'rollback_f1': rollback_f1,
            'rollback_auc': rollback_auc,
            'original_baseline_accuracy': original_baseline_accuracy,
            'accuracy_degradation': accuracy_degradation,
            'degradation_percentage': (accuracy_degradation / original_baseline_accuracy) * 100,
        }
        
        print(f"Original baseline accuracy: {original_baseline_accuracy:.3f}")
        print(f"Rollback accuracy on drift: {rollback_accuracy:.3f}")
        print(f"Performance degradation: {accuracy_degradation:.3f} ({results['degradation_percentage']:.1f}%)")
        
        self.intervention_results['rollback'] = results
        return self.baseline_model, results
    

def run_intervention_testing():
    print("Starting Interventions")
    
    tester = DriftInterventions()
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("base-interventions")
    
    # Load baseline and generate drift
    if not tester.load_baseline_and_drift():
        return
    
    with mlflow.start_run(run_name="basic_interventions"):
        
        # Test each intervention
        print("\\n Testing Interventions:")
        
        # run retraining
        retrained_model, retraining_results = tester.test_intervention_retraining()
        
        # run rollback  
        rollback_model, rollback_results = tester.test_intervention_rollback()
        
        # Log results to MLflow
        print("\\n Logging results to MLflow.")
        
        for intervention_name, results in tester.intervention_results.items():
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{intervention_name}_{metric_name}", value)
                else:
                    mlflow.log_param(f"{intervention_name}_{metric_name}", str(value))
        
        # Save results
        results_fp = "intervention_results.json"
        with open(results_fp, "w") as f:
            json.dump({
                'intervention_results': tester.intervention_results
            }, f, indent=2, default=str)

        try:
            if os.path.exists(results_fp):
                mlflow.log_artifact("intervention_results.json")
                print('Artifact logged')
            else:
                print('File not found')
        except Exception as e:
            print(f'Unable to log artifact: {e}. Results saved locally as {results_fp}')
        
        print("\\n Tests Complete")
        print("\\n Summary:")
        print(f"   • {len(tester.intervention_results)} interventions tested")
        print(f"   • Results logged to MLflow experiment")
        
        return tester.intervention_results

if __name__ == "__main__":
    try:
        results = run_intervention_testing()
        print("\\n Intervention testing completed")
        
    except Exception as e:
        print(f"\\n Intervention testing failed: {e}")