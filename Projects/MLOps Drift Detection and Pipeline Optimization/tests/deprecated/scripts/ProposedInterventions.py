import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time
import json
from pathlib import Path

# Import your existing classes
from DriftDetection import DriftAnalysis

class DriftInterventionsV2:
    """
    Advanced intervention strategies based on academic research
    Includes DDLS active learning, outlier detection, and adaptive ensembles
    """
    
    def __init__(self):
        self.baseline_model = None
        self.baseline_data = None
        self.drift_data = None
        self.drift_labels = None
        self.intervention_results = {}
        self.ddls_regions = None  # Data Distributions with Low Accuracy
        
    def load_baseline_and_drift(self):
        """Load baseline and generate drift data"""
        print("📁 Loading baseline and drift data for advanced interventions...")
        
        drift_analyzer = DriftAnalysis()
        
        if not drift_analyzer.load_baseline():
            return False
            
        self.drift_data, self.drift_labels = drift_analyzer.create_drift_simulation(0.8)
        
        if self.drift_data is None:
            return False
            
        self.baseline_data = drift_analyzer.baseline_data
        self.baseline_model = drift_analyzer.baseline_model
        
        print(f"   ✅ Ready for advanced interventions: {self.drift_data.shape}")
        return True
    
    def identify_ddls_regions(self):
        """
        INTERVENTION 4: Identify Data Distributions with Low Accuracy (DDLS)
        Based on Dong et al. approach using decision trees
        """
        print("\\n🎯 Identifying DDLS regions (Data Distributions with Low Accuracy)...")
        
        # Get baseline model predictions on drift data
        baseline_predictions = self.baseline_model.predict(self.drift_data)
        prediction_probabilities = self.baseline_model.predict_proba(self.drift_data)
        
        # Calculate prediction confidence (max probability)
        prediction_confidence = np.max(prediction_probabilities, axis=1)
        
        # Identify low accuracy instances
        incorrect_predictions = (baseline_predictions != self.drift_labels)
        low_confidence = prediction_confidence < 0.7  # Threshold for low confidence
        
        # Combine criteria for DDLS identification
        ddls_mask = incorrect_predictions | low_confidence
        
        print(f"   📊 Total instances: {len(self.drift_data)}")
        print(f"   🎯 Incorrect predictions: {incorrect_predictions.sum()} ({incorrect_predictions.mean()*100:.1f}%)")
        print(f"   📉 Low confidence: {low_confidence.sum()} ({low_confidence.mean()*100:.1f}%)")
        print(f"   🚨 DDLS regions: {ddls_mask.sum()} ({ddls_mask.mean()*100:.1f}%)")
        
        # Train decision tree to characterize DDLS regions
        ddls_tree = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_leaf=50, 
            random_state=42
        )
        
        # Features for DDLS identification (use subset for interpretability)
        ddls_features = self.drift_data.iloc[:, :8]  # Use first 8 features for simplicity
        ddls_tree.fit(ddls_features, ddls_mask)
        
        # Get feature importance for DDLS characterization
        feature_importance = ddls_tree.feature_importances_
        important_features = ddls_features.columns[feature_importance > 0.1]
        
        self.ddls_regions = {
            'mask': ddls_mask,
            'tree_model': ddls_tree,
            'important_features': important_features.tolist(),
            'accuracy_in_ddls': accuracy_score(self.drift_labels[ddls_mask], baseline_predictions[ddls_mask]),
            'accuracy_outside_ddls': accuracy_score(self.drift_labels[~ddls_mask], baseline_predictions[~ddls_mask])
        }
        
        print(f"   ✅ DDLS characterization complete!")
        print(f"   📊 Accuracy in DDLS regions: {self.ddls_regions['accuracy_in_ddls']:.3f}")
        print(f"   📊 Accuracy outside DDLS: {self.ddls_regions['accuracy_outside_ddls']:.3f}")
        print(f"   🔍 Key DDLS features: {important_features.tolist()}")
        
        return self.ddls_regions
    
    def test_ddls_active_learning(self, labeling_budget=0.2):
        """
        INTERVENTION 5: DDLS-based Active Learning
        Only label instances in low-accuracy regions
        """
        print(f"\\n🎯 Testing DDLS Active Learning (budget: {labeling_budget*100:.0f}%)...")
        
        if self.ddls_regions is None:
            self.identify_ddls_regions()
        
        start_time = time.time()
        
        # Focus labeling on DDLS regions
        ddls_mask = self.ddls_regions['mask']
        ddls_indices = np.where(ddls_mask)[0]
        
        # Calculate labeling budget
        total_budget = int(labeling_budget * len(self.drift_data))
        ddls_budget = min(total_budget, len(ddls_indices))
        
        print(f"   📊 Total labeling budget: {total_budget} instances")
        print(f"   🎯 DDLS instances available: {len(ddls_indices)}")
        print(f"   📝 Will label: {ddls_budget} DDLS instances")
        
        # Select instances for labeling (prioritize by prediction uncertainty)
        if ddls_budget < len(ddls_indices):
            # Select most uncertain instances within DDLS
            ddls_probabilities = self.baseline_model.predict_proba(self.drift_data.iloc[ddls_indices])
            ddls_uncertainty = 1 - np.max(ddls_probabilities, axis=1)
            
            # Select top uncertain instances
            top_uncertain_idx = np.argsort(ddls_uncertainty)[-ddls_budget:]
            selected_indices = ddls_indices[top_uncertain_idx]
        else:
            selected_indices = ddls_indices
        
        # Simulate labeling (in practice, these would be sent to human annotators)
        labeled_X = self.drift_data.iloc[selected_indices]
        labeled_y = self.drift_labels[selected_indices]
        
        # Retrain model on baseline data + selected labeled instances
        if hasattr(self.baseline_model, 'named_steps'):
            # Pipeline model
            from sklearn.pipeline import Pipeline
            
            # Combine baseline training data with new labeled data
            # Note: In practice, you'd need to access original training data
            # Here we simulate by using a portion of drift data as "historical"
            
            retrained_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            
            ddls_retrained_pipeline = Pipeline([
                ('preprocessor', self.baseline_model.named_steps['preprocessor']),
                ('classifier', retrained_model)
            ])
            
            # Train on labeled instances
            ddls_retrained_pipeline.fit(labeled_X, labeled_y)
            
            # Test on remaining drift data
            test_indices = np.setdiff1d(range(len(self.drift_data)), selected_indices)
            test_X = self.drift_data.iloc[test_indices]
            test_y = self.drift_labels[test_indices]
            
            ddls_predictions = ddls_retrained_pipeline.predict(test_X)
            
        else:
            # Direct model
            retrained_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            retrained_model.fit(labeled_X, labeled_y)
            
            test_indices = np.setdiff1d(range(len(self.drift_data)), selected_indices)
            test_X = self.drift_data.iloc[test_indices]
            test_y = self.drift_labels[test_indices]
            
            ddls_predictions = retrained_model.predict(test_X)
        
        training_time = time.time() - start_time
        
        # Evaluate DDLS active learning performance
        ddls_accuracy = accuracy_score(test_y, ddls_predictions)
        ddls_f1 = f1_score(test_y, ddls_predictions)
        
        # Compare with baseline and full retraining
        baseline_predictions = self.baseline_model.predict(test_X)
        baseline_accuracy = accuracy_score(test_y, baseline_predictions)
        
        # Calculate efficiency metrics
        labeling_efficiency = (ddls_accuracy - baseline_accuracy) / (labeling_budget + 1e-8)
        
        results = {
            'intervention_type': 'ddls_active_learning',
            'ddls_accuracy': ddls_accuracy,
            'ddls_f1': ddls_f1,
            'baseline_accuracy': baseline_accuracy,
            'improvement': ddls_accuracy - baseline_accuracy,
            'labeling_budget_used': labeling_budget,
            'instances_labeled': ddls_budget,
            'labeling_efficiency': labeling_efficiency,
            'training_time': training_time,
            'ddls_coverage': len(ddls_indices) / len(self.drift_data)
        }
        
        print(f"   ✅ DDLS Active Learning complete! Time: {training_time:.1f}s")
        print(f"   📊 Baseline accuracy: {baseline_accuracy:.3f}")
        print(f"   📊 DDLS AL accuracy: {ddls_accuracy:.3f}")
        print(f"   📈 Improvement: {ddls_accuracy - baseline_accuracy:+.3f}")
        print(f"   💰 Labeling efficiency: {labeling_efficiency:.3f} per % budget")
        
        self.intervention_results['ddls_active_learning'] = results
        return results
    
    def test_outlier_detection_adjustment(self):
        """
        INTERVENTION 6: Outlier Detection & Dataset Adjustment
        Remove or down-weight outliers that might cause drift
        """
        print("\\n🔍 Testing Outlier Detection & Dataset Adjustment...")
        
        start_time = time.time()
        
        # Apply multiple outlier detection methods
        outlier_methods = {}
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_outliers = iso_forest.fit_predict(self.drift_data.select_dtypes(include=[np.number]))
        outlier_methods['isolation_forest'] = (iso_outliers == -1)
        
        # 2. Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_outliers = lof.fit_predict(self.drift_data.select_dtypes(include=[np.number]))
        outlier_methods['local_outlier_factor'] = (lof_outliers == -1)
        
        # 3. Statistical outliers (Z-score > 3)
        numeric_data = self.drift_data.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
        stat_outliers = (z_scores > 3).any(axis=1)
        outlier_methods['statistical'] = stat_outliers
        
        # Combine outlier detection methods (majority voting)
        outlier_votes = np.stack([outlier_methods[method] for method in outlier_methods])
        consensus_outliers = (outlier_votes.sum(axis=0) >= 2)  # At least 2 methods agree
        
        print(f"   📊 Isolation Forest outliers: {outlier_methods['isolation_forest'].sum()} ({outlier_methods['isolation_forest'].mean()*100:.1f}%)")
        print(f"   📊 LOF outliers: {outlier_methods['local_outlier_factor'].sum()} ({outlier_methods['local_outlier_factor'].mean()*100:.1f}%)")
        print(f"   📊 Statistical outliers: {outlier_methods['statistical'].sum()} ({outlier_methods['statistical'].mean()*100:.1f}%)")
        print(f"   🎯 Consensus outliers: {consensus_outliers.sum()} ({consensus_outliers.mean()*100:.1f}%)")
        
        # Create adjusted dataset by removing consensus outliers
        clean_indices = ~consensus_outliers
        adjusted_X = self.drift_data[clean_indices]
        adjusted_y = self.drift_labels[clean_indices]
        
        # Retrain model on cleaned data
        if hasattr(self.baseline_model, 'named_steps'):
            from sklearn.pipeline import Pipeline
            
            outlier_adjusted_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            
            adjusted_pipeline = Pipeline([
                ('preprocessor', self.baseline_model.named_steps['preprocessor']),
                ('classifier', outlier_adjusted_model)
            ])
            
            adjusted_pipeline.fit(adjusted_X, adjusted_y)
            
            # Test on all drift data (including outliers)
            adjusted_predictions = adjusted_pipeline.predict(self.drift_data)
            
        else:
            outlier_adjusted_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            outlier_adjusted_model.fit(adjusted_X, adjusted_y)
            adjusted_predictions = outlier_adjusted_model.predict(self.drift_data)
        
        processing_time = time.time() - start_time
        
        # Evaluate outlier adjustment performance
        adjusted_accuracy = accuracy_score(self.drift_labels, adjusted_predictions)
        adjusted_f1 = f1_score(self.drift_labels, adjusted_predictions)
        
        # Compare with baseline
        baseline_predictions = self.baseline_model.predict(self.drift_data)
        baseline_accuracy = accuracy_score(self.drift_labels, baseline_predictions)
        
        results = {
            'intervention_type': 'outlier_detection_adjustment',
            'adjusted_accuracy': adjusted_accuracy,
            'adjusted_f1': adjusted_f1,
            'baseline_accuracy': baseline_accuracy,
            'improvement': adjusted_accuracy - baseline_accuracy,
            'outliers_removed': consensus_outliers.sum(),
            'outlier_percentage': consensus_outliers.mean() * 100,
            'data_retention': clean_indices.mean() * 100,
            'processing_time': processing_time,
            'outlier_methods_used': list(outlier_methods.keys())
        }
        
        print(f"   ✅ Outlier adjustment complete! Time: {processing_time:.1f}s")
        print(f"   📊 Baseline accuracy: {baseline_accuracy:.3f}")
        print(f"   📊 Adjusted accuracy: {adjusted_accuracy:.3f}")
        print(f"   📈 Improvement: {adjusted_accuracy - baseline_accuracy:+.3f}")
        print(f"   🗑️ Data retained: {clean_indices.mean()*100:.1f}%")
        
        self.intervention_results['outlier_detection'] = results
        return results
    
    def test_adaptive_ensemble(self):
        """
        INTERVENTION 7: Adaptive Ensemble with Weighted Voting
        Combine multiple models with performance-based weighting
        """
        print("\\n🎪 Testing Adaptive Ensemble with Weighted Voting...")
        
        start_time = time.time()
        
        # Create multiple models with different strategies
        ensemble_models = {}
        model_predictions = {}
        model_weights = {}
        
        # 1. Baseline model
        baseline_pred = self.baseline_model.predict(self.drift_data)
        baseline_accuracy = accuracy_score(self.drift_labels, baseline_pred)
        ensemble_models['baseline'] = self.baseline_model
        model_predictions['baseline'] = baseline_pred
        model_weights['baseline'] = baseline_accuracy
        
        # 2. Retrained model (from previous tests if available)
        if 'retraining' in self.intervention_results:
            # Use retrained model performance
            retrained_accuracy = self.intervention_results['retraining']['retrained_accuracy']
            model_weights['retrained'] = retrained_accuracy
            print(f"   📊 Using previous retraining results: {retrained_accuracy:.3f}")
        else:
            # Create simple retrained model
            X_train, X_test, y_train, y_test = train_test_split(
                self.drift_data, self.drift_labels, test_size=0.3, random_state=42
            )
            
            retrained_model = RandomForestClassifier(n_estimators=50, random_state=42)
            if hasattr(self.baseline_model, 'named_steps'):
                from sklearn.pipeline import Pipeline
                retrained_pipeline = Pipeline([
                    ('preprocessor', self.baseline_model.named_steps['preprocessor']),
                    ('classifier', retrained_model)
                ])
                retrained_pipeline.fit(X_train, y_train)
                retrained_pred = retrained_pipeline.predict(self.drift_data)
            else:
                retrained_model.fit(X_train, y_train)
                retrained_pred = retrained_model.predict(self.drift_data)
            
            retrained_accuracy = accuracy_score(self.drift_labels, retrained_pred)
            ensemble_models['retrained'] = retrained_pipeline if hasattr(self.baseline_model, 'named_steps') else retrained_model
            model_predictions['retrained'] = retrained_pred
            model_weights['retrained'] = retrained_accuracy
        
        # 3. Conservative model (higher regularization)
        conservative_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=20, random_state=42
        )
        
        if hasattr(self.baseline_model, 'named_steps'):
            from sklearn.pipeline import Pipeline
            conservative_pipeline = Pipeline([
                ('preprocessor', self.baseline_model.named_steps['preprocessor']),
                ('classifier', conservative_model)
            ])
            # Train on a subset to make it more conservative
            sample_size = int(0.7 * len(self.drift_data))
            sample_indices = np.random.choice(len(self.drift_data), sample_size, replace=False)
            conservative_pipeline.fit(
                self.drift_data.iloc[sample_indices], 
                self.drift_labels[sample_indices]
            )
            conservative_pred = conservative_pipeline.predict(self.drift_data)
        else:
            sample_size = int(0.7 * len(self.drift_data))
            sample_indices = np.random.choice(len(self.drift_data), sample_size, replace=False)
            conservative_model.fit(
                self.drift_data.iloc[sample_indices], 
                self.drift_labels[sample_indices]
            )
            conservative_pred = conservative_model.predict(self.drift_data)
        
        conservative_accuracy = accuracy_score(self.drift_labels, conservative_pred)
        ensemble_models['conservative'] = conservative_pipeline if hasattr(self.baseline_model, 'named_steps') else conservative_model
        model_predictions['conservative'] = conservative_pred
        model_weights['conservative'] = conservative_accuracy
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        normalized_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        # Create weighted ensemble predictions
        ensemble_predictions = np.zeros(len(self.drift_data))
        
        for model_name, predictions in model_predictions.items():
            ensemble_predictions += normalized_weights[model_name] * predictions
        
        # Convert to binary predictions (threshold at 0.5)
        final_predictions = (ensemble_predictions > 0.5).astype(int)
        
        ensemble_time = time.time() - start_time
        
        # Evaluate ensemble performance
        ensemble_accuracy = accuracy_score(self.drift_labels, final_predictions)
        ensemble_f1 = f1_score(self.drift_labels, final_predictions)
        
        results = {
            'intervention_type': 'adaptive_ensemble',
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1': ensemble_f1,
            'baseline_accuracy': baseline_accuracy,
            'improvement': ensemble_accuracy - baseline_accuracy,
            'model_weights': normalized_weights,
            'individual_accuracies': model_weights,
            'ensemble_time': ensemble_time,
            'models_combined': len(ensemble_models)
        }
        
        print(f"   ✅ Adaptive ensemble complete! Time: {ensemble_time:.1f}s")
        print(f"   📊 Model weights:")
        for model_name, weight in normalized_weights.items():
            accuracy = model_weights[model_name]
            print(f"      {model_name}: {weight:.3f} (acc: {accuracy:.3f})")
        print(f"   📊 Ensemble accuracy: {ensemble_accuracy:.3f}")
        print(f"   📈 Improvement over baseline: {ensemble_accuracy - baseline_accuracy:+.3f}")
        
        self.intervention_results['adaptive_ensemble'] = results
        return results
    
    def compare_advanced_interventions(self):
        """Compare all advanced intervention strategies"""
        print("\\n📊 ADVANCED INTERVENTION COMPARISON")
        print("=" * 50)
        
        if not self.intervention_results:
            print("   ❌ No intervention results to compare!")
            return {}
        
        comparison = {}
        
        for intervention_name, results in self.intervention_results.items():
            if intervention_name == 'ddls_active_learning':
                effectiveness = results['improvement']
                efficiency = results['labeling_efficiency']
                cost = results['labeling_budget_used']
                complexity = 0.6  # Medium complexity
                
            elif intervention_name == 'outlier_detection':
                effectiveness = results['improvement']
                efficiency = effectiveness / (results['processing_time'] + 1e-8)
                cost = 0.1  # Low cost (just computation)
                complexity = 0.3  # Low complexity
                
            elif intervention_name == 'adaptive_ensemble':
                effectiveness = results['improvement']
                efficiency = effectiveness / (results['ensemble_time'] + 1e-8)
                cost = 0.4  # Medium cost (multiple models)
                complexity = 0.5  # Medium complexity
                
            else:
                continue
            
            # Calculate overall score
            overall_score = (
                effectiveness * 0.4 +          # 40% effectiveness
                efficiency * 0.3 +             # 30% efficiency
                (1 - cost) * 0.2 +            # 20% cost (inverted)
                (1 - complexity) * 0.1        # 10% simplicity (inverted)
            )
            
            comparison[intervention_name] = {
                'effectiveness_score': effectiveness,
                'efficiency_score': efficiency,
                'cost_score': cost,
                'complexity_score': complexity,
                'overall_score': overall_score
            }
        
        # Rank interventions
        ranked_interventions = sorted(
            comparison.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        print("\\n🏆 ADVANCED INTERVENTION RANKINGS:")
        for i, (intervention, scores) in enumerate(ranked_interventions, 1):
            print(f"   {i}. {intervention.upper().replace('_', ' ')}")
            print(f"      Effectiveness: {scores['effectiveness_score']:+.3f}")
            print(f"      Efficiency: {scores['efficiency_score']:.3f}")
            print(f"      Cost: {scores['cost_score']:.2f} (lower better)")
            print(f"      Complexity: {scores['complexity_score']:.2f} (lower better)")
            print(f"      Overall Score: {scores['overall_score']:.3f}")
        
        # Provide detailed recommendation
        best_intervention = ranked_interventions[0][0]
        print(f"\\n💡 ADVANCED RECOMMENDATION: {best_intervention.upper().replace('_', ' ')}")
        
        if best_intervention == 'ddls_active_learning':
            print("   ✅ DDLS Active Learning provides targeted labeling efficiency")
            print("   ⚠️ Requires domain expertise for labeling selected instances")
            print("   🎯 Best for scenarios where labeling budget is limited")
            
        elif best_intervention == 'outlier_detection':
            print("   ✅ Outlier detection is fast and low-cost")
            print("   ⚠️ May remove valuable edge cases")
            print("   🎯 Best for scenarios with noisy or corrupted data")
            
        elif best_intervention == 'adaptive_ensemble':
            print("   ✅ Ensemble provides robust performance")
            print("   ⚠️ Higher computational overhead")
            print("   🎯 Best for production systems requiring stability")
        
        return comparison

def run_advanced_intervention_testing():
    """
    Main function to test all advanced drift interventions
    """
    print("🚀 Starting Advanced Drift Intervention Testing")
    print("=" * 55)
    
    tester = AdvancedDriftInterventions()
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-advanced-drift-interventions")
    
    # Load data
    if not tester.load_baseline_and_drift():
        return
    
    with mlflow.start_run(run_name="advanced_interventions_comparison"):
        
        print("\\n🧪 TESTING ADVANCED INTERVENTIONS:")
        
        # Test advanced interventions
        tester.identify_ddls_regions()
        tester.test_ddls_active_learning(labeling_budget=0.15)  # 15% labeling budget
        tester.test_outlier_detection_adjustment()
        tester.test_adaptive_ensemble()
        
        # Compare interventions
        comparison_results = tester.compare_advanced_interventions()
        
        # Log results to MLflow
        print("\\n📝 Logging advanced intervention results to MLflow...")
        
        for intervention_name, results in tester.intervention_results.items():
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"adv_{intervention_name}_{metric_name}", value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"adv_{intervention_name}_{metric_name}_{sub_key}", sub_value)
                else:
                    mlflow.log_param(f"adv_{intervention_name}_{metric_name}", str(value))
        
        # Log comparison results
        if comparison_results:
            for intervention_name, scores in comparison_results.items():
                mlflow.log_metric(f"adv_{intervention_name}_overall_score", scores['overall_score'])
        
        # Save detailed results
        with open("advanced_intervention_results.json", "w") as f:
            json.dump({
                'intervention_results': tester.intervention_results,
                'comparison_scores': comparison_results,
                'ddls_analysis': tester.ddls_regions
            }, f, indent=2, default=str)
        
        mlflow.log_artifact("advanced_intervention_results.json")
        
        print("\\n🎉 Advanced intervention testing complete!")
        print("\\n📊 Summary:")
        print(f"   • {len(tester.intervention_results)} advanced interventions tested")
        print(f"   • DDLS regions identified and characterized")
        print(f"   • Outlier detection and dataset adjustment performed")
        print(f"   • Adaptive ensemble with weighted voting implemented")
        
        print("\\n🔗 Comprehensive Results:")
        print("   1. Review MLflow UI: http://localhost:5000")
        print("   2. Compare with basic interventions")
        print("   3. Select optimal strategy for production deployment")
        
        return tester.intervention_results, comparison_results

if __name__ == "__main__":
    try:
        results = run_advanced_intervention_testing()
        print("\\n✅ Advanced intervention testing completed successfully!")
        print("\\n🎯 You now have a comprehensive intervention strategy comparison!")
        
    except Exception as e:
        print(f"\\n❌ Advanced intervention testing failed: {e}")
        print("\\nTroubleshooting:")
        print("   • Ensure baseline and basic interventions completed")
        print("   • Check required packages: scikit-learn, isolation forest")
        print("   • Verify drift detection framework is working")