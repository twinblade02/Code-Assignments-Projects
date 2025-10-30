import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

from DriftDetection import DriftAnalysis

class ModelComparison:
    def __init__(self):
        self.original_analyzer = DriftAnalysis()
        self.robust_analyzer = DriftAnalysis()
        self.results = {}
        self.plot_data = {}
    
    def setup(self):
        print("=== Setting up Model Comparison Experiment ===")
    
        print("Loading original baseline")
        mlflow.set_tracking_uri("http://localhost:5000")
        original_loaded = self.original_analyzer.load_baseline()
        
        print("Loading robust baseline")
        robust_loaded = self.load_robust_baseline()
        
        if not original_loaded or not robust_loaded:
            print("Failed to load one or both models")
            return False
        
        print("Both models loaded successfully")
        return True
    
    def load_robust_baseline(self):
        try:
            # Load robust training data
            self.robust_analyzer.baseline_data = pd.read_csv("robust_training_data.csv")
            if 'Churn' in self.robust_analyzer.baseline_data.columns:
                self.robust_analyzer.baseline_data = self.robust_analyzer.baseline_data.drop('Churn', axis=1)
            
            # Load robust feature metadata
            with open("feature_metadata_robust.json", 'r') as f:
                metadata = json.load(f)  
            self.robust_analyzer.numeric_features = metadata['numeric_features']
            self.robust_analyzer.categorical_features = metadata['categorical_features']
            
            # Load robust model
            self.robust_analyzer.baseline_model = mlflow.sklearn.load_model("models:/telco-robust-baseline/latest")
            
            print("Robust baseline loaded")
            return True
        except Exception as e:
            print(f"Failed to load robust baseline: {e}")
            return False
    
    def run_drift_strength_analysis(self, drift_strengths=[0.2, 0.4, 0.6, 0.8, 1.0]):
        print(f"\n=== Running Drift Strength Analysis ===")
        
        original_performance = {'accuracy': [], 'f1': [], 'auc': []}
        robust_performance = {'accuracy': [], 'f1': [], 'auc': []}
        
        for strength in drift_strengths:
            print(f"\nTesting drift strength: {strength}")
            
            # Generate drift data
            original_drift_data, original_labels = self.original_analyzer.create_drift_simulation(strength)
            robust_drift_data, robust_labels = self.robust_analyzer.create_drift_simulation(strength)
            
            # Test original model
            original_pred = self.original_analyzer.baseline_model.predict(original_drift_data)
            original_prob = self.original_analyzer.baseline_model.predict_proba(original_drift_data)[:, 1]
            
            original_acc = accuracy_score(original_labels, original_pred)
            original_f1 = f1_score(original_labels, original_pred)
            original_auc = roc_auc_score(original_labels, original_prob)
            
            original_performance['accuracy'].append(original_acc)
            original_performance['f1'].append(original_f1)
            original_performance['auc'].append(original_auc)
            
            # Test robust model
            robust_pred = self.robust_analyzer.baseline_model.predict(robust_drift_data)
            robust_prob = self.robust_analyzer.baseline_model.predict_proba(robust_drift_data)[:, 1]
            
            robust_acc = accuracy_score(robust_labels, robust_pred)
            robust_f1 = f1_score(robust_labels, robust_pred)
            robust_auc = roc_auc_score(robust_labels, robust_prob)
            
            robust_performance['accuracy'].append(robust_acc)
            robust_performance['f1'].append(robust_f1)
            robust_performance['auc'].append(robust_auc)
            
            print(f"   Original: Acc={original_acc:.3f}, F1={original_f1:.3f}, AUC={original_auc:.3f}")
            print(f"   Robust:   Acc={robust_acc:.3f}, F1={robust_f1:.3f}, AUC={robust_auc:.3f}")
        
        # Store for plotting
        self.plot_data = {
            'drift_strengths': drift_strengths,
            'original_performance': original_performance,
            'robust_performance': robust_performance
        }
        
        return drift_strengths, original_performance, robust_performance
    
    def run_single_comparison(self, drift_strength=0.5):
        print(f"\n=== Running Detailed Comparison (drift strength: {drift_strength}) ===")
        
        # Generate drift data
        original_drift_data, original_labels = self.original_analyzer.create_drift_simulation(drift_strength)
        robust_drift_data, robust_labels = self.robust_analyzer.create_drift_simulation(drift_strength)
        
        if original_drift_data is None or robust_drift_data is None:
            print("Failed to create drift data")
            return None
        
        # Test models and get detailed results
        original_pred = self.original_analyzer.baseline_model.predict(original_drift_data)
        original_prob = self.original_analyzer.baseline_model.predict_proba(original_drift_data)[:, 1]
        
        robust_pred = self.robust_analyzer.baseline_model.predict(robust_drift_data)
        robust_prob = self.robust_analyzer.baseline_model.predict_proba(robust_drift_data)[:, 1]
        
        # Calculate metrics
        original_accuracy = accuracy_score(original_labels, original_pred)
        original_f1 = f1_score(original_labels, original_pred)
        original_auc = roc_auc_score(original_labels, original_prob)
        
        robust_accuracy = accuracy_score(robust_labels, robust_pred)
        robust_f1 = f1_score(robust_labels, robust_pred)
        robust_auc = roc_auc_score(robust_labels, robust_prob)
        
        # Store detailed results for plotting
        self.plot_data['detailed'] = {
            'original_labels': original_labels,
            'original_pred': original_pred,
            'original_prob': original_prob,
            'robust_labels': robust_labels,
            'robust_pred': robust_pred,
            'robust_prob': robust_prob
        }
        
        # Calculate improvements
        accuracy_improvement = robust_accuracy - original_accuracy
        f1_improvement = robust_f1 - original_f1
        auc_improvement = robust_auc - original_auc
        
        print(f"   Original Model - Accuracy: {original_accuracy:.3f}, F1: {original_f1:.3f}, AUC: {original_auc:.3f}")
        print(f"   Robust Model - Accuracy: {robust_accuracy:.3f}, F1: {robust_f1:.3f}, AUC: {robust_auc:.3f}")
        print(f"   Improvements: Acc={accuracy_improvement:+.3f}, F1={f1_improvement:+.3f}, AUC={auc_improvement:+.3f}")
        
        # Determine winner
        #if accuracy_improvement > 0.02:
        #    winner = "robust"
        #    print("Robust model performs better!")
        #elif accuracy_improvement < -0.02:
        #    winner = "original"
        #    print("Original model performs better!")
        #else:
        #    winner = "tie"
        #    print("Models perform similarly")
        
        self.results = {
            'original_accuracy': original_accuracy,
            'original_f1': original_f1,
            'original_auc': original_auc,
            'robust_accuracy': robust_accuracy,
            'robust_f1': robust_f1,
            'robust_auc': robust_auc,
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement,
            'auc_improvement': auc_improvement,
            'drift_strength': drift_strength
        }
        
        return self.results
    
    def create_performance_plots(self):
        print("\n=== Creating Performance Plots ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Performance vs Drift Strength
        ax1 = plt.subplot(2, 3, 1)
        drift_strengths = self.plot_data['drift_strengths']
        
        plt.plot(drift_strengths, self.plot_data['original_performance']['accuracy'], 
                'o-', label='Original Model', linewidth=2, markersize=8, color='#FF6B6B')
        plt.plot(drift_strengths, self.plot_data['robust_performance']['accuracy'], 
                'o-', label='Robust Model', linewidth=2, markersize=8, color='#4ECDC4')
        
        plt.xlabel('Drift Strength')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy vs Drift Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.0)
        
        # Plot 2: F1 Score vs Drift Strength
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(drift_strengths, self.plot_data['original_performance']['f1'], 
                'o-', label='Original Model', linewidth=2, markersize=8, color='#FF6B6B')
        plt.plot(drift_strengths, self.plot_data['robust_performance']['f1'], 
                'o-', label='Robust Model', linewidth=2, markersize=8, color='#4ECDC4')
        
        plt.xlabel('Drift Strength')
        plt.ylabel('F1 Score')
        plt.title('Model F1 Score vs Drift Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.3, 0.8)
        
        # Plot 3: AUC vs Drift Strength
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(drift_strengths, self.plot_data['original_performance']['auc'], 
                'o-', label='Original Model', linewidth=2, markersize=8, color='#FF6B6B')
        plt.plot(drift_strengths, self.plot_data['robust_performance']['auc'], 
                'o-', label='Robust Model', linewidth=2, markersize=8, color='#4ECDC4')
        
        plt.xlabel('Drift Strength')
        plt.ylabel('AUC')
        plt.title('Model AUC vs Drift Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.0)
        
        # Plot 4: Performance Improvement
        ax4 = plt.subplot(2, 3, 4)
        accuracy_improvements = [r - o for r, o in zip(self.plot_data['robust_performance']['accuracy'], 
                                                      self.plot_data['original_performance']['accuracy'])]
        f1_improvements = [r - o for r, o in zip(self.plot_data['robust_performance']['f1'], 
                                                self.plot_data['original_performance']['f1'])]
        auc_improvements = [r - o for r, o in zip(self.plot_data['robust_performance']['auc'], 
                                                 self.plot_data['original_performance']['auc'])]
        
        plt.plot(drift_strengths, accuracy_improvements, 'o-', label='Accuracy Improvement', 
                linewidth=2, markersize=8, color='#45B7D1')
        plt.plot(drift_strengths, f1_improvements, 'o-', label='F1 Improvement', 
                linewidth=2, markersize=8, color='#96CEB4')
        plt.plot(drift_strengths, auc_improvements, 'o-', label='AUC Improvement', 
                linewidth=2, markersize=8, color='#FFEAA7')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Drift Strength')
        plt.ylabel('Performance Improvement')
        plt.title('Robust Model Improvement Over Original')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: ROC Curves (if detailed data available)
        if 'detailed' in self.plot_data:
            ax5 = plt.subplot(2, 3, 5)
            
            # Original model ROC
            fpr_orig, tpr_orig, _ = roc_curve(self.plot_data['detailed']['original_labels'], 
                                            self.plot_data['detailed']['original_prob'])
            auc_orig = roc_auc_score(self.plot_data['detailed']['original_labels'], 
                                   self.plot_data['detailed']['original_prob'])
            
            # Robust model ROC
            fpr_robust, tpr_robust, _ = roc_curve(self.plot_data['detailed']['robust_labels'], 
                                                self.plot_data['detailed']['robust_prob'])
            auc_robust = roc_auc_score(self.plot_data['detailed']['robust_labels'], 
                                     self.plot_data['detailed']['robust_prob'])
            
            plt.plot(fpr_orig, tpr_orig, label=f'Original Model (AUC = {auc_orig:.3f})', 
                    linewidth=2, color='#FF6B6B')
            plt.plot(fpr_robust, tpr_robust, label=f'Robust Model (AUC = {auc_robust:.3f})', 
                    linewidth=2, color='#4ECDC4')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Confusion Matrix Comparison
        if 'detailed' in self.plot_data:
            ax6 = plt.subplot(2, 3, 6)
            
            # Create side-by-side confusion matrices
            cm_orig = confusion_matrix(self.plot_data['detailed']['original_labels'], 
                                     self.plot_data['detailed']['original_pred'])
            cm_robust = confusion_matrix(self.plot_data['detailed']['robust_labels'], 
                                       self.plot_data['detailed']['robust_pred'])
            
            # Normalize confusion matrices
            cm_orig_norm = cm_orig.astype('float') / cm_orig.sum(axis=1)[:, np.newaxis]
            cm_robust_norm = cm_robust.astype('float') / cm_robust.sum(axis=1)[:, np.newaxis]
            
            # Create a combined plot
            fig2, (ax_orig, ax_robust) = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.heatmap(cm_orig_norm, annot=True, fmt='.3f', cmap='Reds', 
                       ax=ax_orig, xticklabels=['No Churn', 'Churn'], 
                       yticklabels=['No Churn', 'Churn'])
            ax_orig.set_title('Original Model\nConfusion Matrix')
            ax_orig.set_xlabel('Predicted')
            ax_orig.set_ylabel('Actual')
            
            sns.heatmap(cm_robust_norm, annot=True, fmt='.3f', cmap='Blues', 
                       ax=ax_robust, xticklabels=['No Churn', 'Churn'], 
                       yticklabels=['No Churn', 'Churn'])
            ax_robust.set_title('Robust Model\nConfusion Matrix')
            ax_robust.set_xlabel('Predicted')
            ax_robust.set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance plots created and saved")
    
    def create_drift_robustness_summary(self):
        print("Creating drift robustness summary...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate average performance degradation
        drift_strengths = self.plot_data['drift_strengths']
        
        # Calculate degradation from no-drift baseline (assume strength 0.2 is baseline)
        orig_baseline = self.plot_data['original_performance']['accuracy'][0]  # First point
        robust_baseline = self.plot_data['robust_performance']['accuracy'][0]
        
        orig_degradation = [orig_baseline - acc for acc in self.plot_data['original_performance']['accuracy']]
        robust_degradation = [robust_baseline - acc for acc in self.plot_data['robust_performance']['accuracy']]
        
        # Create bar plot
        x = np.arange(len(drift_strengths))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_degradation, width, label='Original Model', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, robust_degradation, width, label='Robust Model', 
                      color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Drift Strength')
        ax.set_ylabel('Accuracy Degradation')
        ax.set_title('Model Robustness: Accuracy Degradation Under Drift')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:.1f}' for s in drift_strengths])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('drift_robustness_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Drift robustness summary created")
    
    def log_results_with_plots(self):
        print("\n=== Logging Results and Plots to MLflow ===")
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("telco-pipeline-comparison")
        
        with mlflow.start_run(run_name="original-vs-robust-with-plots"):
            # Log metrics
            if self.results:
                mlflow.log_metric("original_accuracy", self.results['original_accuracy'])
                mlflow.log_metric("original_f1", self.results['original_f1'])
                mlflow.log_metric("original_auc", self.results['original_auc'])
                mlflow.log_metric("robust_accuracy", self.results['robust_accuracy'])
                mlflow.log_metric("robust_f1", self.results['robust_f1'])
                mlflow.log_metric("robust_auc", self.results['robust_auc'])
                mlflow.log_metric("accuracy_improvement", self.results['accuracy_improvement'])
                mlflow.log_metric("f1_improvement", self.results['f1_improvement'])
                mlflow.log_metric("auc_improvement", self.results['auc_improvement'])
                #mlflow.log_param("experiment_winner", self.results['winner'])
            
            # Log drift strength analysis results
            for i, strength in enumerate(self.plot_data['drift_strengths']):
                mlflow.log_metric(f"original_acc_drift_{strength}", 
                                self.plot_data['original_performance']['accuracy'][i])
                mlflow.log_metric(f"robust_acc_drift_{strength}", 
                                self.plot_data['robust_performance']['accuracy'][i])
                mlflow.log_metric(f"improvement_drift_{strength}", 
                                self.plot_data['robust_performance']['accuracy'][i] - 
                                self.plot_data['original_performance']['accuracy'][i])
            
            # Log plot artifacts
            plot_files = [
                'model_performance_comparison.png',
                'confusion_matrices_comparison.png', 
                'drift_robustness_summary.png'
            ]
            
            for plot_file in plot_files:
                if Path(plot_file).exists():
                    mlflow.log_artifact(plot_file)
                    print(f"   Logged {plot_file}")
            
            # Log experiment parameters
            mlflow.log_param("comparison_type", "original_vs_robust")
            mlflow.log_param("drift_strengths_tested", str(self.plot_data['drift_strengths']))
            mlflow.log_param("plots_included", True)
            
        print("Results and plots logged to MLflow successfully!")

def run_model_comparison_with_plots():
    print("Starting Model Comparison Experiment with Visualization...")
    
    # Initialize experiment
    experiment = ModelComparison()
    
    # Setup models
    if not experiment.setup():
        return None
    
    # Run drift strength analysis
    experiment.run_drift_strength_analysis(drift_strengths=[0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Run detailed single comparison
    results = experiment.run_single_comparison(drift_strength=0.5)
    if results is None:
        return None
    
    # Create all plots
    experiment.create_performance_plots()
    experiment.create_drift_robustness_summary()
    
    # Log everything to MLflow
    experiment.log_results_with_plots()
    
    print("\n Model comparison complete")
    return results

if __name__ == "__main__":
    try:
        results = run_model_comparison_with_plots()
        if results:
            print("Results are available on MLFlow UI")
        else:
            print("Experiment failed")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
