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
        self.numeric_features = None
        self.categorical_features = None
        
    def load_baseline(self, experiment_name="telco-baseline"):
        print("Loading baseline data and model from MLflow")
        
        # connection
        mlflow.set_tracking_uri("http://localhost:5000")
        
        try:
            # Load and error handling if unreadable
            self.baseline_data = pd.read_csv("baseline_raw_data-V2.csv")
            if 'Churn' in self.baseline_data.columns:
                print('Target detected - removing from loaded data')
                self.baseline_data = self.baseline_data.drop('Churn', axis=1)
            print(f"Baseline data loaded: {self.baseline_data.shape}")

            with open("feature_metadata.json", "r") as f:
                metadata = json.load(f)

            self.numeric_features = metadata['numeric_features']
            self.categorical_features = metadata['categoric_features']
            
            # model load
            self.baseline_model = mlflow.sklearn.load_model("models:/telco_churn_baseline@baselinev3")
            print(f"Model loaded: {type(self.baseline_model).__name__}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load from MLflow: {e}")
            print("Troubleshoot from error message")
            return False
    
    def create_drift_simulation(self, strength=0.8):
        print(f"Creating drift simulation (strength: {strength})")
        
        if self.baseline_data is None:
            print("No baseline data loaded!")
            return None, None
        
        np.random.seed(42)  # le reproducibility
        drift_data = self.baseline_data.copy()

        drift_changes = []
        
        # This is for the numeric features - AI suggestion instead of using a generator; going to see if this works or not
        for i, col in enumerate(self.numeric_features):
            if i % 4 == 0:
                factor = 1 + strength + np.random.normal(0, 0.1, len(drift_data))
                drift_data[col] = drift_data[col] + factor
                drift_changes.append(f'{col}: {strength*100:.0f}% mean increase')

            elif i % 4 == 1:
                noise = np.random.normal(0, strength * drift_data[col].std(), len(drift_data))
                drift_data[col] = drift_data[col] + noise
                drift_changes.append(f"{col}: {strength*100:.0f}% variance increase")

            elif i % 4 == 2:
                drift_data[col] = drift_data[col].max() + drift_data[col].min() - drift_data[col]
                drift_changes.append(f"{col}: Distribution inverted")

            else:
                outlier_mask = np.random.binomial(1, 0.15, len(drift_data)).astype(bool)
                drift_data.loc[outlier_mask, col] = drift_data.loc[outlier_mask, col] * (1 + strength * 3)
                drift_changes.append(f"{col}: 15% extreme outliers added")

        # This is for categoric features - AI suggestion
        for i, col in enumerate(self.categorical_features):
            if col in drift_data.columns:
                
                unique_vals = drift_data[col].unique()
                
                if len(unique_vals) >= 2:
                    
                    if i % 3 == 0:
                        # Category redistribution - shift probabilities
                        mask = np.random.binomial(1, strength, len(drift_data)).astype(bool)
                        # Flip categories to simulate errors
                        if len(unique_vals) == 2:
                            # Binary categorical - flip values
                            val1, val2 = unique_vals[0], unique_vals[1]
                            swap_vals = drift_data.loc[mask, col].map({val1: val2, val2: val1})
                            drift_data.loc[mask, col] = swap_vals
                        else:
                            # Multi-category handling
                            current_vals = drift_data.loc[mask, col].values
                            new_vals = []
                            for val in current_vals:
                                avail = [val for val in unique_vals if val != val]
                                if avail:
                                    new_vals.append(np.random.choice(avail))
                                else:
                                    new_vals.append(val)
                            #new_vals = np.random.choice(unique_vals, size=len(current_vals)) #this throws an error
                            drift_data.loc[mask, col] = new_vals
                        
                        drift_changes.append(f"{col}: {strength*100:.0f}% category redistribution")
                    
                    elif i % 3 == 1:
                        # Introduce new category bias
                        # Heavily bias towards one category
                        most_common = drift_data[col].mode()[0]
                        bias_mask = np.random.binomial(1, strength * 0.6, len(drift_data)).astype(bool)
                        drift_data.loc[bias_mask, col] = most_common
                        drift_changes.append(f"{col}: {strength*60:.0f}% bias towards '{most_common}'")
                    
                    else:
                        # Complete category replacement for subset
                        replace_mask = np.random.binomial(1, strength * 0.4, len(drift_data)).astype(bool)
                        if len(unique_vals) >= 2:
                            # Replace with least common category
                            least_common = drift_data[col].value_counts().index[-1]
                            drift_data.loc[replace_mask, col] = least_common
                            drift_changes.append(f"{col}: {strength*40:.0f}% replaced with '{least_common}'")
        
        # from V1, adapted for this to simulate external factors - some of them
        if strength >= 0.7:
            print("Adding cross-feature relationship drift")
            
            # Create artificial correlations between features
            if len(self.numeric_features) >= 2:
                feat1, feat2 = self.numeric_features[:2]
                correlation_mask = drift_data[feat1] > drift_data[feat1].median()
                drift_data.loc[correlation_mask, feat2] = drift_data.loc[correlation_mask, feat2] * 1.5
                drift_changes.append(f"Cross-feature: {feat1} now affects {feat2}")
        
        # label generation
        base_churn_rate = 0.27  # Original Telco churn rate, see dataset information
        drift_churn_rate = base_churn_rate + (strength * 0.3)
        
        # Make churn rate dependent on some drifted features
        churn_probabilities = np.full(len(drift_data), drift_churn_rate)
        
        # Simulating high churn for high charges
        if 'MonthlyCharges' in drift_data.columns:
            high_charges = drift_data['MonthlyCharges'] > drift_data['MonthlyCharges'].quantile(0.8)
            churn_probabilities[high_charges] += 0.2
        
        # Generate labels
        drift_labels = np.random.binomial(1, np.clip(churn_probabilities, 0.1, 0.9))
        
        print(f"Drift applied to {len(drift_changes)} features:")
        for change in drift_changes[:8]:
            print(f"    {change}")
        if len(drift_changes) > 8:
            print(f"and {len(drift_changes) - 8} more changes")
        
        print(f"Original churn rate: ~27%, Drift churn rate: {drift_labels.mean():.1%}")
        
        
        #if drift_type == "covariate":
        #    for col in numeric_cols[:5]:
        #        shift = np.random.normal(strength, 0.1, len(drift_data)) # this should be a bit simpler I think
        #        drift_data[col] = drift_data[col] * (1 + shift)
        
        #elif drift_type == "concept":
        #    for col in numeric_cols[:5]:
        #        noise = np.random.normal(0, strength * drift_data[col].std(), len(drift_data))
        #        drift_data[col] = drift_data[col] + noise
        
        # Need to simulate synthetic labels too - COMMENTED OUT TO TEST ABOVE REFACTOR
        #drift_labels = np.random.binomial(1, 0.35, len(drift_data))
        
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
            
            print("Drift report created")
            
            report_dict = self.drift_report.as_dict()
            drift_results = self._extract_drift_summary(report_dict)
            
            return drift_results
            
        except Exception as e:
            print(f"Evidently analysis failed: {e}")
            return self._fallback_analysis(drift_data)
    
    def run_tests(self, drift_data):
        print("Running Evidently test suite")
        
        self.drift_test_suite = TestSuite(tests=[
            TestNumberOfDriftedColumns(gte=0, lte=len(self.baseline_data.columns)),
            TestShareOfDriftedColumns(gte=0, lte=0.8)
        ])
        
        try:
            self.drift_test_suite.run(
                reference_data=self.baseline_data,
                current_data=drift_data
            )
            
            # get results
            test_results = self.drift_test_suite.as_dict()
        
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
            
            from sklearn.metrics import accuracy_score, f1_score
            
            accuracy = accuracy_score(drift_labels, predictions)
            f1 = f1_score(drift_labels, predictions)
            
            baseline_accuracy = 0.76  # had to set this from baseline experiment; TODO: Change each time model performance changes
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
        print("Generating output")
        
        # lets just see if this works before trying to do a dashboard, need to save it because the environment doesn't allow a two way connection
        # It does, keep for V2
        if self.drift_report:
            html_path = "evidently_report.html"
            self.drift_report.save_html(html_path)
            print(f"HTML report saved: {html_path}")
        
        return "evidently_report.html"
    
    def _extract_drift_summary(self, report_dict):
        try:
            # Navigate Evidently's report structure
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    
                    drift_by_columns = result.get('drift_by_columns', {})
                    drifted_features = [col for col, info in drift_by_columns.items() if info.get('drift_detected', False)]

                    numeric_drifted = [n for n in drifted_features if n in self.numeric_features]
                    categoric_drifted = [c for c in drifted_features if c in self.categorical_features]
                    
                    return {
                        'summary': {
                            'dataset_drift': result.get('dataset_drift', False),
                            'drift_score': result.get('drift_score', 0.0),
                            'drifted_features_count': len(drifted_features),
                            'total_features': len(drift_by_columns),
                            'drift_percentage': (len(drifted_features) / len(drift_by_columns)) * 100 if drift_by_columns else 0,
                            'drifted_features': drifted_features
                        },
                        'drift_by_type': {
                            'numeric_drifted': len(numeric_drifted),
                            'categorical_drifted': len(categoric_drifted),
                            'numeric_drifted_features': numeric_drifted,
                            'categorical_drifted_features': categoric_drifted
                        },
                        'results': drift_by_columns
                    }
            
            return {'summary': {'dataset_drift': False, 'error': 'Unable to parse results'}}
            
        except Exception as e:
            return {'error': f'Extraction failed: {e}'}

    def _fallback_analysis(self, drift_data):
        print("Using fallback analysis.")
        
        from scipy.stats import ks_2samp, chi2_contingency
        
        drift_results = []
        
        for col in self.numeric_features:
            if col in self.baseline_data.columns and col in drift_data.columns:
                ks_stat, p_value = ks_2samp(self.baseline_data[col], drift_data[col])
                drift_results.append({
                    'feature': col,
                    'type': 'numeric',
                    'test': 'ks_test',
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                })
        
        for col in self.categorical_features:
            if col in self.baseline_data.columns and col in drift_data.columns:
                try:
                    # Chi-square test for categorical features
                    baseline_counts = self.baseline_data[col].value_counts()
                    drift_counts = drift_data[col].value_counts()
                    
                    all_categories = set(baseline_counts.index) | set(drift_counts.index)
                    baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
                    drift_aligned = [drift_counts.get(cat, 0) for cat in all_categories]
                    
                    chi2_stat, p_value, _, _ = chi2_contingency([baseline_aligned, drift_aligned])
                    
                    drift_results.append({
                        'feature': col,
                        'type': 'categorical', 
                        'test': 'chi2_test',
                        'statistic': chi2_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05
                    })
                    
                except Exception as e:
                    print(f" Chi2 test failed for {col}: {e}")
        
        drifted_features = [r['feature'] for r in drift_results if r['drift_detected']]
        numeric_drifted = [r['feature'] for r in drift_results if r['type'] == 'numeric' and r['drift_detected']]
        categorical_drifted = [r['feature'] for r in drift_results if r['type'] == 'categorical' and r['drift_detected']]
        
        return {
            'summary': {
                'dataset_drift': len(drifted_features) > 0,
                'drift_score': np.mean([r['statistic'] for r in drift_results]),
                'drifted_features_count': len(drifted_features),
                'total_features': len(drift_results),
                'drift_percentage': (len(drifted_features) / len(drift_results)) * 100,
                'drifted_features': drifted_features
            },
            'drift_by_type': {
                'numeric_drifted': len(numeric_drifted),
                'categorical_drifted': len(categorical_drifted),
                'numeric_drifted_features': numeric_drifted,
                'categorical_drifted_features': categorical_drifted
            },
            'method': 'fallback_statistical_tests',
            'detailed_results': drift_results
        }

    def create_robust_features_for_drift(self, drift_data):
        print("Creating robust features for drift simulation")
    
        robust_drift = drift_data.copy()
        features_added = []

        #if 'SeniorCitizen' not in robust_drift.columns:
        #    print("Adding missing SeniorCitizen column")
        #    robust_drift['SeniorCitizen'] = np.random.choice([0, 1], size=len(robust_drift), p=[0.84, 0.16])
        #    print("Added SeniorCitizen")
      
        #if 'OnlineBackup' not in robust_drift.columns:
        #   print("Adding missing OnlineBackup column")
        #    robust_drift['OnlineBackup'] = np.random.choice(['Yes', 'No', 'No internet service'], size=len(robust_drift), p=[0.35, 0.45, 0.20])
        #    print("Added OnlineBackup")
    
        if 'MonthlyCharges' in robust_drift.columns and 'TotalCharges' in robust_drift.columns:
            robust_drift['monthly_total_ratio'] = robust_drift['MonthlyCharges'] / (robust_drift['TotalCharges'] + 1)
            features_added.append('monthly_total_ratio')
            print("Added monthly_total_ratio")
    
        if 'TotalCharges' in robust_drift.columns and 'tenure' in robust_drift.columns:
            robust_drift['charge_per_month'] = robust_drift['TotalCharges'] / (robust_drift['tenure'] + 1)
            features_added.append('charge_per_month')
            print("Added charge_per_month")
    
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [col for col in service_cols if col in robust_drift.columns]
    
        if available_services:
            service_count = 0
            for col in available_services:
                service_count += (robust_drift[col] == 'Yes').astype(int)
            robust_drift['service_engagement'] = service_count
            features_added.append('service_engagement')
            print(f"Added service_engagement from {len(available_services)} services")
    
        if 'tenure' in robust_drift.columns:
            robust_drift['tenure_tier'] = pd.qcut(robust_drift['tenure'], 
                                                q=5, labels=['New', 'Short', 'Medium', 'Long', 'Veteran'], 
                                                duplicates='drop').astype(str)
            features_added.append('tenure_tier')
            print("Added tenure_tier")
    
        if 'MonthlyCharges' in robust_drift.columns:
            robust_drift['value_tier'] = pd.qcut(robust_drift['MonthlyCharges'], 
                                                q=4, labels=['Budget', 'Standard', 'Premium', 'Enterprise'], 
                                            duplicates='drop').astype(str)
            features_added.append('value_tier')
            print("Added value_tier")
    
        stability_score = 0
        if 'Contract' in robust_drift.columns:
            stability_score += (robust_drift['Contract'] == 'Two year').astype(int) * 2
            stability_score += (robust_drift['Contract'] == 'One year').astype(int) * 1
    
        if 'PaymentMethod' in robust_drift.columns:
            auto_pay = robust_drift['PaymentMethod'].str.contains('automatic', case=False, na=False)
            stability_score += auto_pay.astype(int)
    
        robust_drift['stability_score'] = stability_score
        features_added.append('stability_score')
        print("Added stability_score")
    
        print(f"Total robust features added: {len(features_added)}")
        print(f"Robust drift data shape: {robust_drift.shape}")
    
        return robust_drift

    def create_drift_simulation_with_robust_features(self, strength):
        print(f"Creating drift simulation with robust features (strength: {strength})")
    
    # Use your existing drift simulation method
        drift_data, drift_labels = self.create_drift_simulation(strength)
    
        if drift_data is None:
            print("Failed to create base drift simulation")
            return None, None
    
    # Add robust features to drift data
        robust_drift_data = self.create_robust_features_for_drift(drift_data)
    
        print(f"Robust drift simulation complete: {robust_drift_data.shape}")
    
        return robust_drift_data, drift_labels

def run_complete_analysis():

    print("Starting Evidently Analysis")
    
    # Initialize analyzer
    analyzer = DriftAnalysis()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-driftAnalysis")
    
    with mlflow.start_run(run_name="evidently_analysis_V2"):
        
        if not analyzer.load_baseline():
            print("Cannot proceed without baseline data. Run baseline experiment first!")
            return
        
        drift_data, drift_labels = analyzer.create_drift_simulation(0.8)
        if drift_data is None:
            return
        
        drift_results = analyzer.run_analysis(drift_data, 0.1)
        
        test_results = analyzer.run_tests(drift_data)
        
        impact_results = analyzer.measure_impact(drift_data, drift_labels)
        
        html_report = analyzer.get_output(drift_results, test_results, impact_results)
        
        if drift_results and 'summary' in drift_results:
            mlflow.log_metric("drift_detected", 1 if drift_results.get('dataset_drift') else 0)
            mlflow.log_metric("drift_percentage", drift_results.get('drift_percentage', 0))
            mlflow.log_metric("drifted_features_count", drift_results.get('drifted_features_count', 0))

            if 'drift_by_type' in drift_results:
                type_info = drift_results['drift_by_type']
                mlflow.log_metric('numeric_features_drifted', type_info.get('numeric_drifted', 0))
                mlflow.log_metric('categorical_features_drifted', type_info.get('categorical_drifted', 0))
        
        if test_results:
            mlflow.log_metric("tests_passed", test_results.get('passed_tests', 0))
            mlflow.log_metric("tests_total", test_results.get('total_tests', 0))
        
        if impact_results:
            mlflow.log_metric("accuracy_drop_percentage", impact_results.get('accuracy_drop_percentage', 0))
            mlflow.log_metric("drift_accuracy", impact_results.get('drift_accuracy', 0))
            #mlflow.log_metric("additional_errors", impact_results.get('additional_errors', 0)) # Terminal is for errors, not artifacts
        
        # Dropping this artifact because PSQL doesn't like this
        #if Path("evidently_report.html").exists():
        #    mlflow.log_artifact("evidently_report.html")
        
        print("\\n Analysis Complete")
        print("Key metrics logged")
        
        return analyzer, drift_results, test_results, impact_results

if __name__ == "__main__":
    try:
        results = run_complete_analysis()
        print("\\ Analysis completed")
        
    except Exception as e:
        print(f"\\n Analysis failed: {e}. See tracebacks")