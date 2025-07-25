import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, log_loss
)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.evaluation_results = {}
        
    def load_models(self, models_dir='models/'):
        """Load trained models and preprocessing objects"""
        print("Loading trained models...")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Loaded scaler")
        
        # Load feature columns
        features_path = os.path.join(models_dir, 'feature_columns.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            print(f"✓ Loaded {len(self.feature_columns)} feature columns")
        
        # Load individual models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title().replace(' ', '_')
            model_path = os.path.join(models_dir, model_file)
            
            try:
                self.models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model with 8+ comprehensive metrics"""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate 8 key metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan,
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
                'log_loss': log_loss(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
            }
            
            # Additional metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            metrics.update({
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Same as recall
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Same as precision
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'balanced_accuracy': (metrics['sensitivity'] + metrics['specificity']) / 2
            })
            
            # Confusion matrix values
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            })
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all loaded models"""
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Prepare test data based on model type
            if model_name in ['Random_Forest', 'Gradient_Boosting', 'Xgboost', 'Decision_Tree', 'Adaboost']:
                X_test_model = X_test  # Use original features
            else:
                X_test_model = self.scaler.transform(X_test) if self.scaler else X_test
            
            metrics = self.evaluate_model(model, X_test_model, y_test, model_name)
            
            if metrics:
                results[model_name] = metrics
                print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            else:
                print(f"  ✗ Evaluation failed")
        
        self.evaluation_results = results
        return results
    
    def create_evaluation_summary(self):
        """Create comprehensive evaluation summary"""
        print("\n=== EVALUATION SUMMARY ===")
        
        if not self.evaluation_results:
            print("No evaluation results available")
            return None
        
        # Create DataFrame with all metrics
        metrics_df = pd.DataFrame(self.evaluation_results).T
        
        # Round numerical columns
        numerical_cols = metrics_df.select_dtypes(include=[np.number]).columns
        metrics_df[numerical_cols] = metrics_df[numerical_cols].round(4)
        
        # Sort by ROC-AUC score
        metrics_df = metrics_df.sort_values('roc_auc', ascending=False)
        
        print("Model Performance Ranking (by ROC-AUC):")
        print("=" * 80)
        
        # Print top performers
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cohen_kappa', 'matthews_corrcoef']
        
        for i, (model_name, row) in enumerate(metrics_df.iterrows()):
            print(f"\n{i+1}. {model_name}")
            print("-" * 40)
            for metric in key_metrics:
                if metric in row and not pd.isna(row[metric]):
                    print(f"   {metric.replace('_', ' ').title()}: {row[metric]:.4f}")
        
        # Business impact analysis
        print(f"\n=== BUSINESS IMPACT ANALYSIS ===")
        
        best_model = metrics_df.index[0]
        best_metrics = metrics_df.iloc[0]
        
        print(f"Best Overall Model: {best_model}")
        print(f"Expected Fraud Detection Rate: {best_metrics['recall']:.1%}")
        print(f"Expected False Positive Rate: {best_metrics['false_positive_rate']:.1%}")
        
        # Calculate potential cost savings (hypothetical)
        avg_fraud_amount = 25000  # Hypothetical average fraud amount
        total_claims = 1000  # Hypothetical monthly claims
        fraud_rate = 0.15  # 15% fraud rate
        
        frauds_detected = total_claims * fraud_rate * best_metrics['recall']
        false_positives = total_claims * (1 - fraud_rate) * best_metrics['false_positive_rate']
        
        savings = frauds_detected * avg_fraud_amount
        investigation_cost = (frauds_detected + false_positives) * 500  # $500 per investigation
        
        net_savings = savings - investigation_cost
        
        print(f"\nHypothetical Monthly Impact (1000 claims):")
        print(f"  - Frauds Detected: {frauds_detected:.0f}")
        print(f"  - False Positives: {false_positives:.0f}")
        print(f"  - Gross Savings: ${savings:,.0f}")
        print(f"  - Investigation Costs: ${investigation_cost:,.0f}")
        print(f"  - Net Savings: ${net_savings:,.0f}")
        
        return metrics_df
    
    def save_evaluation_results(self, output_dir='outputs/'):
        """Save evaluation results to files"""
        print("\n=== SAVING EVALUATION RESULTS ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.evaluation_results:
            # Save detailed metrics as JSON
            results_path = os.path.join(output_dir, 'model_evaluation_metrics.json')
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                serializable_results = {}
                for model, metrics in self.evaluation_results.items():
                    serializable_results[model] = {}
                    for key, value in metrics.items():
                        if isinstance(value, (np.integer, np.floating)):
                            serializable_results[model][key] = value.item()
                        else:
                            serializable_results[model][key] = value
                
                json.dump(serializable_results, f, indent=2)
            print(f"Saved detailed metrics to {results_path}")
            
            # Save summary as CSV
            summary_df = pd.DataFrame(self.evaluation_results).T
            summary_path = os.path.join(output_dir, 'model_evaluation_summary.csv')
            summary_df.to_csv(summary_path)
            print(f"Saved summary to {summary_path}")
            
            # Create model comparison report
            self.create_comparison_report(output_dir)
        
    def create_comparison_report(self, output_dir):
        """Create detailed model comparison report"""
        report_path = os.path.join(output_dir, 'model_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("AUTO INSURANCE FRAUD DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if self.evaluation_results:
                # Best model analysis
                best_model = max(self.evaluation_results.keys(), 
                               key=lambda x: self.evaluation_results[x]['roc_auc'])
                best_metrics = self.evaluation_results[best_model]
                
                f.write(f"Best Performing Model: {best_model}\n")
                f.write(f"ROC-AUC Score: {best_metrics['roc_auc']:.4f}\n")
                f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {best_metrics['precision']:.4f}\n")
                f.write(f"Recall: {best_metrics['recall']:.4f}\n\n")
                
                f.write("DETAILED MODEL COMPARISON\n")
                f.write("-" * 30 + "\n\n")
                
                # Sort models by performance
                sorted_models = sorted(self.evaluation_results.items(), 
                                     key=lambda x: x[1]['roc_auc'], reverse=True)
                
                for i, (model_name, metrics) in enumerate(sorted_models):
                    f.write(f"{i+1}. {model_name}\n")
                    f.write("   " + "-" * 20 + "\n")
                    f.write(f"   Accuracy:         {metrics['accuracy']:.4f}\n")
                    f.write(f"   Precision:        {metrics['precision']:.4f}\n")
                    f.write(f"   Recall:           {metrics['recall']:.4f}\n")
                    f.write(f"   F1-Score:         {metrics['f1_score']:.4f}\n")
                    f.write(f"   ROC-AUC:          {metrics['roc_auc']:.4f}\n")
                    f.write(f"   Cohen's Kappa:    {metrics['cohen_kappa']:.4f}\n")
                    f.write(f"   Matthews Corr:    {metrics['matthews_corrcoef']:.4f}\n")
                    f.write(f"   Specificity:      {metrics['specificity']:.4f}\n")
                    f.write(f"   False Pos Rate:   {metrics['false_positive_rate']:.4f}\n\n")
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                f.write(f"1. Deploy {best_model} as the primary fraud detection model\n")
                f.write("2. Implement ensemble approach combining top 3 models\n")
                f.write("3. Set fraud probability threshold based on business requirements\n")
                f.write("4. Monitor model performance and retrain periodically\n")
        
        print(f"Saved comparison report to {report_path}")

def main():
    """Main evaluation pipeline"""
    print("Starting model evaluation pipeline...")
    
    # Load test data
    try:
        # Load the preprocessed data and split it the same way as training
        df = pd.read_csv('data/processed/engineered_auto_insurance.csv')
        
        # Prepare data the same way as training
        df['Fraud_Binary'] = (df['Fraud_Ind'] == 'Y').astype(int)
        
        feature_columns = [
            'Age_Insured', 'Policy_Premium', 'Policy_Ded', 'Umbrella_Limit',
            'Capital_Gains', 'Capital_Loss', 'Accident_Hour', 'Num_of_Vehicles_Involved',
            'Bodily_Injuries', 'Witnesses', 'Auto_Year', 'Vehicle_Cost', 'Annual_Mileage',
            'DiffIN_Mileage', 'Total_Claim', 'Injury_Claim', 'Property_Claim', 'Vehicle_Claim',
            'Claim_to_VehicleCost_Ratio', 'Premium_to_Claim_Ratio', 'Vehicle_Age',
            'High_Mileage_Flag', 'Claim_Severity_Score'
        ]
        
        # Add temporal features if available
        if 'Claim_Reporting_Delay' in df.columns:
            feature_columns.append('Claim_Reporting_Delay')
        if 'Policy_Tenure_at_Accident' in df.columns:
            feature_columns.append('Policy_Tenure_at_Accident')
        
        # Filter existing features
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].fillna(df[feature_columns].median())
        y = df['Fraud_Binary']
        
        # Use same split as training
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Test fraud rate: {y_test.mean():.2%}")
        
    except FileNotFoundError:
        print("Data files not found. Please run preprocessing and training first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models()
    
    if not evaluator.models:
        print("No models found. Please run model training first.")
        return
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(X_test, y_test)
    
    # Create evaluation summary
    summary_df = evaluator.create_evaluation_summary()
    
    # Save results
    evaluator.save_evaluation_results()
    
    print("\nModel evaluation completed successfully!")
    print(f"Evaluated {len(results)} models")
    print("Results saved to 'outputs/' directory")

if __name__ == "__main__":
    main()
