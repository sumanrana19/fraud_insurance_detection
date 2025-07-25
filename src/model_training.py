import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Fraud_Ind'
        
    def prepare_data(self, df):
        """Prepare data for model training"""
        print("\n=== DATA PREPARATION ===")
        
        # Convert target variable to binary
        df['Fraud_Binary'] = (df[self.target_column] == 'Y').astype(int)
        
        # Select features for modeling
        feature_columns = [
            # Original numerical features
            'Age_Insured', 'Policy_Premium', 'Policy_Ded', 'Umbrella_Limit',
            'Capital_Gains', 'Capital_Loss', 'Accident_Hour', 'Num_of_Vehicles_Involved',
            'Bodily_Injuries', 'Witnesses', 'Auto_Year', 'Vehicle_Cost', 'Annual_Mileage',
            'DiffIN_Mileage', 'Total_Claim', 'Injury_Claim', 'Property_Claim', 'Vehicle_Claim',
            
            # Engineered features
            'Claim_to_VehicleCost_Ratio', 'Premium_to_Claim_Ratio', 'Vehicle_Age',
            'High_Mileage_Flag', 'Claim_Severity_Score'
        ]
        
        # Add temporal features if available
        if 'Claim_Reporting_Delay' in df.columns:
            feature_columns.append('Claim_Reporting_Delay')
        if 'Policy_Tenure_at_Accident' in df.columns:
            feature_columns.append('Policy_Tenure_at_Accident')
        
        # Filter features that exist in the dataset
        self.feature_columns = [col for col in feature_columns if col in df.columns]
        
        print(f"Selected {len(self.feature_columns)} features for modeling")
        print("Features:", self.feature_columns)
        
        # Prepare X and y
        X = df[self.feature_columns].copy()
        y = df['Fraud_Binary'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def initialize_models(self):
        """Initialize all 10 classification models"""
        print("\n=== MODEL INITIALIZATION ===")
        
        self.models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'Neural_Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self, X, y, use_smote=True, test_size=0.2):
        """Train all models with cross-validation"""
        print("\n=== MODEL TRAINING ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for imbalanced dataset
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {X_train_scaled.shape[0]} samples")
            print(f"New target distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        # Convert back to DataFrame for tree-based models
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        # Store processed data
        self.X_train, self.X_test = X_train_df, X_test_df
        self.y_train, self.y_test = y_train, y_test
        
        trained_models = {}
        cv_scores = {}
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for most models, original for tree-based
                if name in ['Random_Forest', 'Gradient_Boosting', 'XGBoost', 'Decision_Tree', 'AdaBoost']:
                    X_train_model = X_train_df
                else:
                    X_train_model = X_train_scaled
                
                # Train model
                model.fit(X_train_model, y_train)
                trained_models[name] = model
                
                # Cross-validation score
                cv_score = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='roc_auc')
                cv_scores[name] = {
                    'mean': cv_score.mean(),
                    'std': cv_score.std(),
                    'scores': cv_score.tolist()
                }
                
                print(f"  ✓ CV ROC-AUC: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
                continue
        
        self.trained_models = trained_models
        self.cv_scores = cv_scores
        
        print(f"\nSuccessfully trained {len(trained_models)} models")
        return trained_models, cv_scores
    
    def save_models(self, models_dir='models/'):
        """Save trained models and preprocessing objects"""
        print("\n=== SAVING MODELS ===")
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.trained_models.items():
            model_path = os.path.join(models_dir, f'{name.lower()}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature columns
        import json
        features_path = os.path.join(models_dir, 'feature_columns.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"Saved feature columns to {features_path}")
        
        # Save cross-validation scores
        cv_path = os.path.join(models_dir, 'cv_scores.json')
        with open(cv_path, 'w') as f:
            json.dump(self.cv_scores, f, indent=2)
        print(f"Saved CV scores to {cv_path}")
    
    def get_feature_importance(self):
        """Extract feature importance from tree-based models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        importance_data = {}
        
        tree_models = ['Random_Forest', 'Gradient_Boosting', 'XGBoost', 'Decision_Tree', 'AdaBoost']
        
        for name in tree_models:
            if name in self.trained_models:
                model = self.trained_models[name]
                
                if hasattr(model, 'feature_importances_'):
                    importance_data[name] = {
                        'features': self.feature_columns,
                        'importance': model.feature_importances_.tolist()
                    }
                    
                    # Print top 10 features
                    feature_imp = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n{name} - Top 10 Features:")
                    print(feature_imp.head(10).to_string(index=False))
        
        return importance_data

def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    
    # Load engineered data
    try:
        df = pd.read_csv('data/processed/engineered_auto_insurance.csv')
        print(f"Loaded engineered data: {len(df)} records")
    except FileNotFoundError:
        print("Engineered data not found. Please run feature engineering first.")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    trained_models, cv_scores = trainer.train_models(X, y)
    
    # Get feature importance
    importance_data = trainer.get_feature_importance()
    
    # Save everything
    trainer.save_models()
    
    # Save feature importance
    if importance_data:
        import json
        os.makedirs('models/', exist_ok=True)
        with open('models/feature_importance.json', 'w') as f:
            json.dump(importance_data, f, indent=2)
        print("Saved feature importance to 'models/feature_importance.json'")
    
    print("\n=== TRAINING SUMMARY ===")
    print(f"Successfully trained {len(trained_models)} models")
    print("Best models by CV ROC-AUC:")
    
    sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
    for i, (name, scores) in enumerate(sorted_scores[:5]):
        print(f"{i+1}. {name}: {scores['mean']:.4f} (+/- {scores['std'] * 2:.4f})")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
