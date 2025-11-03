import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import your existing frameworks
from DriftDetection_Enhanced import EnhancedDriftAnalysis
from enhanced_intervention_testing import AdvancedDriftInterventions

class PipelineRobustnessScorer:
    """
    Comprehensive pipeline robustness evaluation system
    Quantifies intervention effectiveness across multiple drift scenarios
    """
    
    def __init__(self):
        self.baseline_performance = {}
        self.drift_scenarios = {}
        self.intervention_scores = {}
        self.robustness_matrix = None
        
    def generate_drift_scenarios(self):
        """
        Generate multiple drift scenarios with varying strengths and types
        """
        print("🌪️ Generating comprehensive drift scenarios...")
        
        drift_analyzer = EnhancedDriftAnalysis()
        if not drift_analyzer.load_baseline_fixed():
            return False
            
        scenarios = {}
        
        # Define drift scenario matrix
        drift_types = ['covariate', 'concept', 'comprehensive']
        drift_strengths = [0.3, 0.5, 0.8, 1.0]  # Mild to extreme
        
        for drift_type in drift_types:
            for strength in drift_strengths:
                scenario_name = f"{drift_type}_drift_{strength}"
                
                print(f"   📊 Creating {scenario_name}...")
                
                # Generate drift data
                if hasattr(drift_analyzer, 'create_comprehensive_drift_simulation_fixed'):
                    drift_data, drift_labels = drift_analyzer.create_comprehensive_drift_simulation_fixed(
                        drift_type, strength
                    )
                else:
                    # Fallback method name
                    drift_data, drift_labels = drift_analyzer.create_drift_simulation(
                        drift_type, strength
                    )
                
                if drift_data is not None:
                    scenarios[scenario_name] = {
                        'drift_type': drift_type,
                        'strength': strength,
                        'data': drift_data,
                        'labels': drift_labels,
                        'baseline_accuracy': None,  # Will be calculated
                        'drift_characteristics': {
                            'data_shape': drift_data.shape,
                            'label_distribution': drift_labels.mean()
                        }
                    }
        
        self.drift_scenarios = scenarios
        print(f"   ✅ Generated {len(scenarios)} drift scenarios")
        return True
    
    def calculate_baseline_performance(self):
        """
        Calculate baseline model performance on each drift scenario
        """
        print("📊 Calculating baseline performance across drift scenarios...")
        
        drift_analyzer = EnhancedDriftAnalysis()
        if not drift_analyzer.load_baseline_fixed():
            return False
            
        baseline_model = drift_analyzer.baseline_model
        
        for scenario_name, scenario in self.drift_scenarios.items():
            # Test baseline model on this drift scenario
            baseline_predictions = baseline_model.predict(scenario['data'])
            baseline_accuracy = accuracy_score(scenario['labels'], baseline_predictions)
            baseline_f1 = f1_score(scenario['labels'], baseline_predictions)
            
            scenario['baseline_accuracy'] = baseline_accuracy
            scenario['baseline_f1'] = baseline_f1
            
            print(f"   📈 {scenario_name}: Acc={baseline_accuracy:.3f}, F1={baseline_f1:.3f}")
        
        return True
    
    def evaluate_intervention_robustness(self, intervention_name, intervention_func):
        """
        Evaluate a specific intervention across all drift scenarios
        Returns comprehensive robustness scores
        """
        print(f"\\n🎯 Evaluating {intervention_name} robustness...")
        
        intervention_results = {}
        
        for scenario_name, scenario in self.drift_scenarios.items():
            print(f"   🧪 Testing on {scenario_name}...")
            
            start_time = time.time()
            
            try:
                # Apply intervention to this scenario
                result = intervention_func(
                    scenario['data'], 
                    scenario['labels'],
                    scenario['drift_type'],
                    scenario['strength']
                )
                
                intervention_time = time.time() - start_time
                
                # Extract key metrics
                if result:
                    intervention_accuracy = result.get('accuracy', 0)
                    intervention_f1 = result.get('f1_score', 0)
                    computational_cost = result.get('training_time', intervention_time)
                    labeling_cost = result.get('labeling_budget', 0)
                    
                    # Calculate robustness metrics
                    baseline_acc = scenario['baseline_accuracy']
                    
                    # Performance stability (higher is better)
                    performance_recovery = max(0, (intervention_accuracy - baseline_acc) / (1 - baseline_acc + 1e-8))
                    performance_stability = min(100, intervention_accuracy * 100)
                    
                    # Response speed (lower time = higher score)
                    speed_score = max(0, 100 - (computational_cost / 60) * 10)  # Penalty for >6min
                    
                    # Cost efficiency (lower cost = higher score)  
                    cost_score = max(0, 100 - labeling_cost * 100 - (computational_cost > 300) * 20)
                    
                    # Adaptability (consistent performance across scenarios)
                    adaptability_score = min(100, performance_stability * (1 - scenario['strength'] * 0.2))
                    
                    intervention_results[scenario_name] = {
                        'accuracy': intervention_accuracy,
                        'f1_score': intervention_f1,
                        'baseline_accuracy': baseline_acc,
                        'performance_recovery': performance_recovery,
                        'performance_stability_score': performance_stability,
                        'speed_score': speed_score,
                        'cost_score': cost_score,
                        'adaptability_score': adaptability_score,
                        'computational_time': computational_cost,
                        'labeling_cost': labeling_cost,
                        'drift_strength': scenario['strength']
                    }
                    
                    print(f"      ✅ Acc: {intervention_accuracy:.3f}, Recovery: {performance_recovery:.3f}")
                
                else:
                    # Intervention failed
                    intervention_results[scenario_name] = {
                        'accuracy': scenario['baseline_accuracy'],
                        'f1_score': 0,
                        'baseline_accuracy': scenario['baseline_accuracy'],
                        'performance_recovery': 0,
                        'performance_stability_score': 0,
                        'speed_score': 0,
                        'cost_score': 0,
                        'adaptability_score': 0,
                        'computational_time': intervention_time,
                        'labeling_cost': 0,
                        'drift_strength': scenario['strength'],
                        'failed': True
                    }
                    print(f"      ❌ Intervention failed")
            
            except Exception as e:
                print(f"      ❌ Error: {e}")
                intervention_results[scenario_name] = {
                    'accuracy': scenario['baseline_accuracy'],
                    'performance_stability_score': 0,
                    'speed_score': 0,
                    'cost_score': 0,
                    'adaptability_score': 0,
                    'failed': True,
                    'error': str(e)
                }
        
        # Calculate overall robustness score for this intervention
        overall_score = self._calculate_overall_robustness_score(intervention_results)
        
        self.intervention_scores[intervention_name] = {
            'scenario_results': intervention_results,
            'overall_robustness_score': overall_score,
            'performance_consistency': self._calculate_consistency_score(intervention_results, 'performance_stability_score'),
            'speed_consistency': self._calculate_consistency_score(intervention_results, 'speed_score'),
            'cost_effectiveness': np.mean([r.get('cost_score', 0) for r in intervention_results.values()]),
            'adaptability_score': np.mean([r.get('adaptability_score', 0) for r in intervention_results.values()])
        }
        
        print(f"   🏆 Overall Robustness Score: {overall_score:.1f}/100")
        
        return intervention_results
    
    def _calculate_overall_robustness_score(self, intervention_results):
        """Calculate composite robustness score"""
        valid_results = [r for r in intervention_results.values() if not r.get('failed', False)]
        
        if not valid_results:
            return 0
        
        # Weighted average of component scores
        performance_scores = [r['performance_stability_score'] for r in valid_results]
        speed_scores = [r['speed_score'] for r in valid_results]
        cost_scores = [r['cost_score'] for r in valid_results]
        adaptability_scores = [r['adaptability_score'] for r in valid_results]
        
        overall_score = (
            np.mean(performance_scores) * 0.4 +
            np.mean(speed_scores) * 0.25 +
            np.mean(cost_scores) * 0.2 +
            np.mean(adaptability_scores) * 0.15
        )
        
        return overall_score
    
    def _calculate_consistency_score(self, intervention_results, metric_name):
        """Calculate consistency across different drift scenarios"""
        valid_results = [r for r in intervention_results.values() if not r.get('failed', False)]
        
        if len(valid_results) < 2:
            return 0
        
        values = [r.get(metric_name, 0) for r in valid_results]
        # Lower standard deviation = higher consistency
        consistency = max(0, 100 - np.std(values))
        
        return consistency
    
    def create_robustness_matrix(self):
        """
        Create comprehensive robustness comparison matrix
        """
        print("\\n📊 Creating Pipeline Robustness Matrix...")
        
        if not self.intervention_scores:
            print("   ❌ No intervention scores available!")
            return None
        
        # Create comparison matrix
        interventions = list(self.intervention_scores.keys())
        metrics = [
            'overall_robustness_score',
            'performance_consistency', 
            'speed_consistency',
            'cost_effectiveness',
            'adaptability_score'
        ]
        
        matrix_data = []
        
        for intervention in interventions:
            scores = self.intervention_scores[intervention]
            row = [intervention]
            
            for metric in metrics:
                row.append(scores.get(metric, 0))
            
            matrix_data.append(row)
        
        # Create DataFrame
        columns = ['Intervention'] + [m.replace('_', ' ').title() for m in metrics]
        self.robustness_matrix = pd.DataFrame(matrix_data, columns=columns)
        
        # Sort by overall robustness score
        self.robustness_matrix = self.robustness_matrix.sort_values(
            'Overall Robustness Score', ascending=False
        )
        
        print("   ✅ Robustness matrix created!")
        print("\\n🏆 PIPELINE ROBUSTNESS RANKINGS:")
        print(self.robustness_matrix.round(1).to_string(index=False))
        
        return self.robustness_matrix
    
    def generate_robustness_report(self):
        """
        Generate comprehensive robustness analysis report
        """
        print("\\n📋 Generating Pipeline Robustness Report...")
        
        if self.robustness_matrix is None:
            self.create_robustness_matrix()
        
        # Find best intervention overall
        best_intervention = self.robustness_matrix.iloc[0]['Intervention']
        best_score = self.robustness_matrix.iloc[0]['Overall Robustness Score']
        
        # Create detailed report
        report = f"""
PIPELINE ROBUSTNESS ANALYSIS REPORT
==================================

EXECUTIVE SUMMARY:
• {len(self.intervention_scores)} interventions tested across {len(self.drift_scenarios)} drift scenarios
• Best performing intervention: {best_intervention} (Score: {best_score:.1f}/100)
• Drift scenarios ranged from mild (0.3) to extreme (1.0) strength
• Testing covered covariate, concept, and comprehensive drift types

ROBUSTNESS RANKINGS:
"""
        
        for i, row in self.robustness_matrix.iterrows():
            report += f"\\n{i+1}. {row['Intervention'].upper().replace('_', ' ')}"
            report += f"\\n   Overall Score: {row['Overall Robustness Score']:.1f}/100"
            report += f"\\n   Performance Consistency: {row['Performance Consistency']:.1f}/100"
            report += f"\\n   Speed Consistency: {row['Speed Consistency']:.1f}/100"
            report += f"\\n   Cost Effectiveness: {row['Cost Effectiveness']:.1f}/100"
            report += f"\\n   Adaptability: {row['Adaptability Score']:.1f}/100"
        
        # Add recommendations
        report += f"""

RECOMMENDATIONS:

PRODUCTION DEPLOYMENT:
• Primary: {best_intervention.replace('_', ' ').title()}
• Backup: {self.robustness_matrix.iloc[1]['Intervention'].replace('_', ' ').title()}

DRIFT SEVERITY CONSIDERATIONS:
• Mild drift (≤0.5): Focus on speed and cost efficiency
• Severe drift (>0.7): Prioritize performance recovery
• Extreme drift (>0.9): Consider ensemble approaches

MONITORING STRATEGY:
• Set robustness score threshold at {best_score * 0.8:.1f}
• Trigger intervention when score drops below threshold
• Regular robustness assessment every 30 days

COST-BENEFIT ANALYSIS:
• High-robustness interventions justify cost for critical applications
• Medium-robustness suitable for standard business applications
• Low-cost interventions appropriate for experimental systems
"""
        
        # Save report
        with open("pipeline_robustness_report.txt", "w") as f:
            f.write(report)
        
        print("   📄 Report saved: pipeline_robustness_report.txt")
        
        return report

# Define intervention wrapper functions for testing
class InterventionWrappers:
    """
    Wrapper functions to standardize intervention testing interface
    """
    
    @staticmethod
    def retraining_wrapper(drift_data, drift_labels, drift_type, strength):
        """Wrapper for retraining intervention"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Split data for training/testing
            X_train, X_test, y_train, y_test = train_test_split(
                drift_data, drift_labels, test_size=0.3, random_state=42
            )
            
            start_time = time.time()
            
            # Train new model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            training_time = time.time() - start_time
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'labeling_budget': 1.0,  # Full retraining
                'method': 'retraining'
            }
            
        except Exception as e:
            return None
    
    @staticmethod
    def ddls_active_learning_wrapper(drift_data, drift_labels, drift_type, strength):
        """Wrapper for DDLS active learning intervention"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Simulate DDLS approach with reduced labeling budget
            labeling_budget = 0.15  # Only 15% of data
            sample_size = int(len(drift_data) * labeling_budget)
            
            # Select uncertain instances (simplified)
            sample_indices = np.random.choice(len(drift_data), sample_size, replace=False)
            
            X_train = drift_data.iloc[sample_indices]
            y_train = drift_labels[sample_indices]
            
            # Test on remaining data
            test_indices = np.setdiff1d(range(len(drift_data)), sample_indices)
            X_test = drift_data.iloc[test_indices]
            y_test = drift_labels[test_indices]
            
            start_time = time.time()
            
            # Train model on selected data
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            training_time = time.time() - start_time
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'labeling_budget': labeling_budget,
                'method': 'ddls_active_learning'
            }
            
        except Exception as e:
            return None
    
    @staticmethod
    def ensemble_wrapper(drift_data, drift_labels, drift_type, strength):
        """Wrapper for ensemble intervention"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Create multiple models with different strategies
            X_train, X_test, y_train, y_test = train_test_split(
                drift_data, drift_labels, test_size=0.3, random_state=42
            )
            
            start_time = time.time()
            
            # Create ensemble of 3 models
            models = []
            for i in range(3):
                model = RandomForestClassifier(
                    n_estimators=50 + i*25, 
                    max_depth=8 + i*2, 
                    random_state=42+i
                )
                model.fit(X_train, y_train)
                models.append(model)
            
            # Ensemble prediction (majority voting)
            predictions_ensemble = []
            for model in models:
                predictions_ensemble.append(model.predict(X_test))
            
            # Majority vote
            ensemble_pred = np.round(np.mean(predictions_ensemble, axis=0)).astype(int)
            
            accuracy = accuracy_score(y_test, ensemble_pred)
            f1 = f1_score(y_test, ensemble_pred)
            training_time = time.time() - start_time
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'labeling_budget': 1.0,  # Full training for all models
                'method': 'ensemble'
            }
            
        except Exception as e:
            return None

def run_comprehensive_robustness_evaluation():
    """
    Main function to run comprehensive pipeline robustness evaluation
    """
    print("🚀 Starting Comprehensive Pipeline Robustness Evaluation")
    print("=" * 65)
    
    scorer = PipelineRobustnessScorer()
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("telco-pipeline-robustness-evaluation")
    
    with mlflow.start_run(run_name="comprehensive_robustness_analysis"):
        
        # Generate drift scenarios
        if not scorer.generate_drift_scenarios():
            print("❌ Failed to generate drift scenarios")
            return
        
        # Calculate baseline performance
        if not scorer.calculate_baseline_performance():
            print("❌ Failed to calculate baseline performance")
            return
        
        # Test interventions
        interventions_to_test = {
            'retraining': InterventionWrappers.retraining_wrapper,
            'ddls_active_learning': InterventionWrappers.ddls_active_learning_wrapper,
            'ensemble': InterventionWrappers.ensemble_wrapper
        }
        
        print("\\n🧪 TESTING INTERVENTIONS ACROSS ALL DRIFT SCENARIOS:")
        
        for intervention_name, intervention_func in interventions_to_test.items():
            scorer.evaluate_intervention_robustness(intervention_name, intervention_func)
        
        # Create robustness matrix
        robustness_matrix = scorer.create_robustness_matrix()
        
        # Generate comprehensive report
        report = scorer.generate_robustness_report()
        
        # Log results to MLflow
        print("\\n📝 Logging comprehensive robustness results to MLflow...")
        
        # Log overall scores
        for intervention, scores in scorer.intervention_scores.items():
            mlflow.log_metric(f"robustness_{intervention}_overall", scores['overall_robustness_score'])
            mlflow.log_metric(f"robustness_{intervention}_consistency", scores['performance_consistency'])
            mlflow.log_metric(f"robustness_{intervention}_cost_effectiveness", scores['cost_effectiveness'])
            mlflow.log_metric(f"robustness_{intervention}_adaptability", scores['adaptability_score'])
        
        # Log matrix as artifact
        if robustness_matrix is not None:
            robustness_matrix.to_csv("pipeline_robustness_matrix.csv", index=False)
            mlflow.log_artifact("pipeline_robustness_matrix.csv")
        
        # Log report as artifact
        mlflow.log_artifact("pipeline_robustness_report.txt")
        
        # Log scenario results
        with open("robustness_detailed_results.json", "w") as f:
            json.dump({
                'drift_scenarios': {k: {
                    'drift_type': v['drift_type'],
                    'strength': v['strength'],
                    'baseline_accuracy': v['baseline_accuracy'],
                    'data_shape': list(v['data'].shape)
                } for k, v in scorer.drift_scenarios.items()},
                'intervention_scores': scorer.intervention_scores
            }, f, indent=2, default=str)
        
        mlflow.log_artifact("robustness_detailed_results.json")
        
        print("\\n🎉 Comprehensive Pipeline Robustness Evaluation Complete!")
        print("\\n📊 FINAL RESULTS:")
        
        if robustness_matrix is not None:
            best_intervention = robustness_matrix.iloc[0]['Intervention']
            best_score = robustness_matrix.iloc[0]['Overall Robustness Score']
            
            print(f"   🏆 Most Robust Intervention: {best_intervention.replace('_', ' ').title()}")
            print(f"   📈 Robustness Score: {best_score:.1f}/100")
            print(f"   📊 Tested across {len(scorer.drift_scenarios)} drift scenarios")
            print(f"   🎯 {len(interventions_to_test)} interventions compared")
        
        print("\\n📁 Outputs generated:")
        print("   📊 pipeline_robustness_matrix.csv")
        print("   📄 pipeline_robustness_report.txt") 
        print("   📈 MLflow experiment with comprehensive metrics")
        print("   🔗 MLflow UI: http://localhost:5000")
        
        return scorer.intervention_scores, robustness_matrix

if __name__ == "__main__":
    try:
        scores, matrix = run_comprehensive_robustness_evaluation()
        print("\\n✅ Pipeline robustness evaluation completed successfully!")
        print("\\n🎯 You now have quantified robustness scores for all interventions!")
        
    except Exception as e:
        print(f"\\n❌ Robustness evaluation failed: {e}")
        print("\\nTroubleshooting:")
        print("   • Ensure baseline and drift detection frameworks are working")
        print("   • Check MLflow server: http://localhost:5000")
        print("   • Verify all required packages are installed")