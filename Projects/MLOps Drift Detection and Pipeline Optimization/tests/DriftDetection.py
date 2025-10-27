import pandas as pd
import numpy as np
import json
import mlflow
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns, TestColumnDrift
from evidently.tests import TestAccuracyScore, TestF1Score, TestRocAuc

class DriftAnalysis:
    def __init__(self):
        self.baseline_data = None
        self.baseline_model = None
        self.drift_report = None
        self.drift_test_suite = None
        
    def load_baseline(self, experiment_name="telco-baseline"):
        print("Loading baseline data and model from MLflow")
        
        # connection
        mlflow.set_tracking_uri("http://localhost:5000")
        
        try:
            # Load and error handling if unreadable
            self.baseline_data = pd.read_csv("baseline_raw_data-V2.csv")
            print(f"Baseline data loaded: {self.baseline_data.shape}")
            
            # model load
            self.baseline_model = mlflow.sklearn.load_model("models:/telco_churn_baseline/latest")
            print(f"Model loaded: {type(self.baseline_model).__name__}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load from MLflow: {e}")
            print("Make sure you've run the baseline experiment first!")
            return False
    
    def create_drift_simulation(self, drift_type="covariate", strength=0.3):
        print(f"Creating {drift_type} drift simulation (strength: {strength})...")
        
        if self.baseline_data is None:
            print("No baseline data loaded!")
            return None, None
        
        np.random.seed(42)  # le reproducibility
        drift_data = self.baseline_data.copy()
        
        numeric_cols = self.baseline_data.select_dtypes(include=[np.number]).columns
        
        if drift_type == "covariate":
            for col in numeric_cols[:5]:
                shift = np.random.normal(strength, 0.1, len(drift_data)) # this should be a bit simpler I think
                drift_data[col] = drift_data[col] * (1 + shift)
        
        elif drift_type == "concept":
            for col in numeric_cols[:5]:
                noise = np.random.normal(0, strength * drift_data[col].std(), len(drift_data))
                drift_data[col] = drift_data[col] + noise
        
        # Need to simulate synthetic labels too
        drift_labels = np.random.binomial(1, 0.35, len(drift_data))
        
        print(f"Drift data created: {drift_data.shape}")
        print(f"Simulated churn rate: {drift_labels.mean():.1%}")
        
        return drift_data, drift_labels
    
    def run_analysis(self, drift_data, drift_threshold=0.3):
        print("Running drift analysis")
        
        self.drift_report = Report(metrics=[
            DataDriftPreset(drift_share=drift_threshold)
        ])
        
        try:
            self.drift_report.run(
                reference_data=self.baseline_data, 
                current_data=drift_data
            )
            
            print(" Evidently drift report generated successfully!")
            
            report_dict = self.drift_report.as_dict()
            drift_results = self._extract_drift_summary(report_dict)
            
            return drift_results
            
        except Exception as e:
            print(f"Evidently analysis failed: {e}")
            return None
    
    def run_tests(self, drift_data):
        print("Running Evidently test suite...")
        
        self.drift_test_suite = TestSuite(tests=[
            TestNumberOfDriftedColumns(gte=0, lte=5),
            TestShareOfDriftedColumns(gte=0, lte=0.3)
        ])
        
        try:
            self.drift_test_suite.run(
                reference_data=self.baseline_data,
                current_data=drift_data
            )
            
            # get results
            test_results = self.drift_test_suite.as_dict()
            
            # Count passed/failed tests
            tests = test_results.get('tests', [])
            passed_tests = sum(1 for test in tests if test.get('status') == 'SUCCESS')
            total_tests = len(tests)
            
            suite_results = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'overall_pass': passed_tests == total_tests
            }
            
            print(f"Test suite complete: {passed_tests}/{total_tests} tests passed")
            
            return suite_results
            
        except Exception as e:
            print(f"Test suite failed: {e}")
            return None
    
    def measure_impact(self, drift_data, drift_labels):
        print("Measuring impact of drift")
        
        if self.baseline_model is None:
            print(" No model loaded!")
            return None
        
        try:
            # Test model
            predictions = self.baseline_model.predict(drift_data)
            
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            
            accuracy = accuracy_score(drift_labels, predictions)
            f1 = f1_score(drift_labels, predictions)
            
            baseline_accuracy = 0.77  # had to set this from baseline
            accuracy_drop = baseline_accuracy - accuracy
            accuracy_drop_pct = (accuracy_drop / baseline_accuracy) * 100
            
            impact_results = {
                'baseline_accuracy': baseline_accuracy,
                'drift_accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'accuracy_drop_percentage': accuracy_drop_pct,
                'f1_score': f1,
                'predictions_total': len(predictions),
                'additional_errors': int(accuracy_drop * len(predictions))
            }
            
            print(f"Baseline accuracy: {baseline_accuracy:.1%}")
            print(f"Drift accuracy: {accuracy:.1%}")
            print(f"Performance drop: {accuracy_drop_pct:.1f}%")
            print(f"Additional errors: {impact_results['additional_errors']} per {len(predictions)} predictions")
            
            return impact_results
            
        except Exception as e:
            print(f"Impact analysis failed: {e}")
            return None
    
    def get_output(self, drift_results, test_results, impact_results):
        print("Generating stakeholder outputs")
        
        # lets just see if this works before trying to do a dashboard, need to save it because the environment doesn't allow a two way connection
        if self.drift_report:
            html_path = "evidently_report.html"
            self.drift_report.save_html(html_path)
            print(f"Interactive HTML report saved: {html_path}")
        
        return "evidently_report.html"
    
    def _extract_drift_summary(self, report_dict):
        try:
            # Navigate Evidently's report structure
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    
                    drift_by_columns = result.get('drift_by_columns', {})
                    drifted_features = [col for col, info in drift_by_columns.items() 
                                      if info.get('drift_detected', False)]
                    
                    return {
                        'dataset_drift': result.get('dataset_drift', False),
                        'drift_score': result.get('drift_score', 0.0),
                        'drifted_features_count': len(drifted_features),
                        'total_features': len(drift_by_columns),
                        'drift_percentage': (len(drifted_features) / len(drift_by_columns)) * 100 if drift_by_columns else 0,
                        'drifted_features': drifted_features
                    }
            
            return {'error': 'Could not extract drift metrics'}
            
        except Exception as e:
            return {'error': f'Extraction failed: {e}'}

def run_complete_analysis():

    print("Starting Evidently Analysis")
    
    # Initialize analyzer
    analyzer = DriftAnalysis()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-driftAnalysis")
    
    with mlflow.start_run(run_name="evidently_analysis"):
        
        if not analyzer.load_baseline():
            print("Cannot proceed without baseline data. Run baseline experiment first!")
            return
        
        drift_data, drift_labels = analyzer.create_drift_simulation("covariate", 0.5)
        if drift_data is None:
            return
        
        drift_results = analyzer.run_analysis(drift_data, 0.3)
        
        test_results = analyzer.run_tests(drift_data)
        
        impact_results = analyzer.measure_impact(drift_data, drift_labels)
        
        html_report = analyzer.get_output(drift_results, test_results, impact_results)
        
        if drift_results:
            mlflow.log_metric("drift_detected", 1 if drift_results.get('dataset_drift') else 0)
            mlflow.log_metric("drift_percentage", drift_results.get('drift_percentage', 0))
            mlflow.log_metric("drifted_features_count", drift_results.get('drifted_features_count', 0))
        
        if test_results:
            mlflow.log_metric("tests_passed", test_results.get('passed_tests', 0))
            mlflow.log_metric("tests_total", test_results.get('total_tests', 0))
        
        if impact_results:
            mlflow.log_metric("accuracy_drop_percentage", impact_results.get('accuracy_drop_percentage', 0))
            mlflow.log_metric("drift_accuracy", impact_results.get('drift_accuracy', 0))
            mlflow.log_metric("additional_errors", impact_results.get('additional_errors', 0))
        
        if Path("evidently_report.html").exists():
            mlflow.log_artifact("evidently_report.html")
        
        print("\\n Analysis Complete!")
        print("Key metrics logged")
        
        return analyzer, drift_results, test_results, impact_results

if __name__ == "__main__":
    try:
        results = run_complete_analysis()
        print("\\ Analysis completed")
        
    except Exception as e:
        print(f"\\n Analysis failed: {e}. See tracebacks")