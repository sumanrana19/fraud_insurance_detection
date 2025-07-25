import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionUtils:
    """Utility functions for fraud detection system"""
    
    @staticmethod
    def load_config(config_path: str = 'config/config.yaml') -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> None:
        """Save dictionary to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_data = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load dictionary from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"JSON file not found: {filepath}")
            return {}
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return {}
    
    @staticmethod
    def calculate_business_metrics(df: pd.DataFrame, fraud_col: str = 'Fraud_Ind') -> Dict:
        """Calculate business-relevant metrics"""
        metrics = {}
        
        # Basic counts
        total_claims = len(df)
        fraud_claims = (df[fraud_col] == 'Y').sum()
        legitimate_claims = total_claims - fraud_claims
        
        metrics.update({
            'total_claims': total_claims,
            'fraud_claims': fraud_claims,
            'legitimate_claims': legitimate_claims,
            'fraud_rate': fraud_claims / total_claims if total_claims > 0 else 0
        })
        
        # Financial metrics
        if 'Total_Claim' in df.columns:
            fraud_df = df[df[fraud_col] == 'Y']
            legit_df = df[df[fraud_col] == 'N']
            
            metrics.update({
                'total_claim_amount': df['Total_Claim'].sum(),
                'avg_fraud_claim': fraud_df['Total_Claim'].mean() if len(fraud_df) > 0 else 0,
                'avg_legit_claim': legit_df['Total_Claim'].mean() if len(legit_df) > 0 else 0,
                'fraud_loss_amount': fraud_df['Total_Claim'].sum() if len(fraud_df) > 0 else 0
            })
            
            # Loss ratio
            metrics['fraud_loss_ratio'] = (
                metrics['fraud_loss_amount'] / metrics['total_claim_amount'] 
                if metrics['total_claim_amount'] > 0 else 0
            )
        
        return metrics
    
    @staticmethod
    def create_risk_segments(df: pd.DataFrame, 
                           risk_features: List[str],
                           quantile_thresholds: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        """Create risk segments based on multiple features"""
        df_risk = df.copy()
        
        # Calculate composite risk score
        risk_scores = []
        
        for feature in risk_features:
            if feature in df.columns:
                # Normalize feature to 0-1 scale
                feature_norm = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
                risk_scores.append(feature_norm.fillna(0))
        
        if risk_scores:
            # Average risk score
            df_risk['Risk_Score'] = np.mean(risk_scores, axis=0)
            
            # Create segments based on quantiles
            df_risk['Risk_Segment'] = pd.cut(
                df_risk['Risk_Score'],
                bins=[-np.inf] + [df_risk['Risk_Score'].quantile(q) for q in quantile_thresholds] + [np.inf],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        return df_risk
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict:
        """Validate data quality and return quality report"""
        report = {}
        
        # Basic info
        report['shape'] = df.shape
        report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
        
        # Missing values
        missing_counts = df.isnull().sum()
        report['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_percentage': (missing_counts.sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        # Duplicates
        report['duplicates'] = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        report['data_types'] = dtype_counts.to_dict()
        
        # Numerical columns stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_stats'] = {
                'count': len(numeric_cols),
                'zero_values': (df[numeric_cols] == 0).sum().sum(),
                'negative_values': (df[numeric_cols] < 0).sum().sum()
            }
        
        # Categorical columns stats
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            report['categorical_stats'] = {
                'count': len(cat_cols),
                'avg_unique_values': df[cat_cols].nunique().mean(),
                'high_cardinality_cols': (df[cat_cols].nunique() > 100).sum()
            }
        
        return report
    
    @staticmethod
    def generate_feature_summary(df: pd.DataFrame, target_col: str = 'Fraud_Ind') -> pd.DataFrame:
        """Generate comprehensive feature summary"""
        summary_data = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            col_info = {
                'Feature': col,
                'Data_Type': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percentage': (df[col].isnull().sum() / len(df)) * 100,
                'Unique_Values': df[col].nunique(),
                'Cardinality': 'High' if df[col].nunique() > 100 else 'Medium' if df[col].nunique() > 10 else 'Low'
            }
            
            if df[col].dtype in [np.number]:
                col_info.update({
                    'Mean': df[col].mean(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Skewness': df[col].skew(),
                    'Zero_Count': (df[col] == 0).sum()
                })
                
                # Fraud correlation if target exists
                if target_col in df.columns:
                    fraud_mean = df[df[target_col] == 'Y'][col].mean()
                    legit_mean = df[df[target_col] == 'N'][col].mean()
                    col_info['Fraud_Mean'] = fraud_mean
                    col_info['Legit_Mean'] = legit_mean
                    col_info['Mean_Difference'] = fraud_mean - legit_mean
            else:
                # Categorical features
                mode_val = df[col].mode()
                col_info.update({
                    'Mode': mode_val[0] if len(mode_val) > 0 else None,
                    'Mode_Frequency': (df[col] == mode_val[0]).sum() if len(mode_val) > 0 else 0
                })
                
                # Fraud rate by category if target exists
                if target_col in df.columns and len(mode_val) > 0:
                    mode_fraud_rate = (
                        df[(df[col] == mode_val[0]) & (df[target_col] == 'Y')].shape[0] /
                        df[df[col] == mode_val[0]].shape[0]
                    ) if df[df[col] == mode_val[0]].shape[0] > 0 else 0
                    col_info['Mode_Fraud_Rate'] = mode_fraud_rate
            
            summary_data.append(col_info)
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> Dict:
        """Detect outliers in numerical columns"""
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method.lower() == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': outliers.index.tolist()
                }
            
            elif method.lower() == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > factor]
                
                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(df)) * 100,
                    'threshold': factor,
                    'outlier_indices': outliers.index.tolist()
                }
        
        return outlier_info
    
    @staticmethod
    def create_model_comparison_report(model_results: Dict, output_path: str = None) -> str:
        """Create a comprehensive model comparison report"""
        
        if not model_results:
            return "No model results available for comparison."
        
        # Sort models by ROC-AUC
        sorted_models = sorted(
            model_results.items(), 
            key=lambda x: x[1].get('roc_auc', 0), 
            reverse=True
        )
        
        report = []
        report.append("=" * 80)
        report.append("AUTO INSURANCE FRAUD DETECTION - MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append("")
        
        best_model_name, best_metrics = sorted_models[0]
        report.append(f"Best Performing Model: {best_model_name}")
        report.append(f"ROC-AUC Score: {best_metrics.get('roc_auc', 'N/A'):.4f}")
        report.append(f"Accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
        report.append(f"Precision: {best_metrics.get('precision', 'N/A'):.4f}")
        report.append(f"Recall: {best_metrics.get('recall', 'N/A'):.4f}")
        report.append("")
        
        # Detailed Comparison
        report.append("DETAILED MODEL COMPARISON")
        report.append("-" * 30)
        report.append("")
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                      'cohen_kappa', 'matthews_corrcoef']
        
        for i, (model_name, metrics) in enumerate(sorted_models):
            report.append(f"{i+1}. {model_name}")
            report.append("   " + "-" * 30)
            
            for metric in key_metrics:
                if metric in metrics:
                    report.append(f"   {metric.replace('_', ' ').title():<20}: {metrics[metric]:.4f}")
            
            report.append("")
        
        # Performance Analysis
        report.append("PERFORMANCE ANALYSIS")
        report.append("-" * 20)
        report.append("")
        
        # Calculate average metrics
        all_metrics = {}
        for metric in key_metrics:
            values = [m.get(metric, 0) for _, m in sorted_models if metric in m]
            if values:
                all_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        for metric, stats in all_metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}:")
            report.append(f"   Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            report.append(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        report.append("")
        report.append(f"1. Deploy {best_model_name} as the primary fraud detection model")
        report.append("2. Consider ensemble approach combining top 3 performing models")
        report.append("3. Monitor model performance regularly and retrain as needed")
        report.append("4. Implement A/B testing to validate model improvements")
        report.append("5. Set appropriate decision thresholds based on business requirements")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    """Example usage of utility functions"""
    utils = FraudDetectionUtils()
    
    # Load sample data for demonstration
    try:
        df = pd.read_csv('data/processed/clean_auto_insurance.csv')
        print("Loaded sample data for utility demonstration")
        
        # Data quality validation
        quality_report = utils.validate_data_quality(df)
        print("\nData Quality Report:")
        print(f"Shape: {quality_report['shape']}")
        print(f"Missing values: {quality_report['missing_values']['total_missing']}")
        
        # Business metrics
        business_metrics = utils.calculate_business_metrics(df)
        print(f"\nBusiness Metrics:")
        print(f"Fraud rate: {business_metrics['fraud_rate']:.2%}")
        
        # Feature summary
        feature_summary = utils.generate_feature_summary(df)
        print(f"\nGenerated feature summary for {len(feature_summary)} features")
        
        # Outlier detection
        outliers = utils.detect_outliers(df)
        print(f"\nDetected outliers in {len(outliers)} numerical columns")
        
    except FileNotFoundError:
        print("Sample data not found. Please run data preprocessing first.")

if __name__ == "__main__":
    main()
