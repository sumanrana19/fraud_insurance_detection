# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# class FeatureEngineer:
#     def __init__(self):
#         self.feature_names = []
    
#     def create_engineered_features(self, df):
#         """Create five new engineered features"""
#         print("\n=== FEATURE ENGINEERING ===")
        
#         df_eng = df.copy()
        
#         # Feature 1: Claim to Vehicle Cost Ratio
#         df_eng['Claim_to_VehicleCost_Ratio'] = df_eng['Total_Claim'] / (df_eng['Vehicle_Cost'] + 1)
#         print("✓ Created Claim_to_VehicleCost_Ratio")
        
#         # Feature 2: Premium to Claim Ratio
#         df_eng['Premium_to_Claim_Ratio'] = df_eng['Policy_Premium'] / (df_eng['Total_Claim'] + 1)
#         print("✓ Created Premium_to_Claim_Ratio")
        
#         # Feature 3: Vehicle Age
#         current_year = datetime.now().year
#         df_eng['Vehicle_Age'] = current_year - df_eng['Auto_Year']
#         df_eng['Vehicle_Age'] = df_eng['Vehicle_Age'].clip(lower=0)
#         print("✓ Created Vehicle_Age")
        
#         # Feature 4: High Mileage Flag
#         median_mileage = df_eng['Annual_Mileage'].median()
#         df_eng['High_Mileage_Flag'] = (df_eng['Annual_Mileage'] > median_mileage).astype(int)
#         print(f"✓ Created High_Mileage_Flag (threshold: {median_mileage:,.0f} miles)")
        
#         # Feature 5: Claim Severity Score
#         # Weighted composite score: Injury (50%) + Property (30%) + Vehicle (20%)
#         max_injury = df_eng['Injury_Claim'].max()
#         max_property = df_eng['Property_Claim'].max()
#         max_vehicle = df_eng['Vehicle_Claim'].max()
        
#         if max_injury > 0 and max_property > 0 and max_vehicle > 0:
#             df_eng['Claim_Severity_Score'] = (
#                 (df_eng['Injury_Claim'] / max_injury) * 0.5 + 
#                 (df_eng['Property_Claim'] / max_property) * 0.3 + 
#                 (df_eng['Vehicle_Claim'] / max_vehicle) * 0.2
#             )
#         else:
#             df_eng['Claim_Severity_Score'] = 0
        
#         print("✓ Created Claim_Severity_Score")
        
#         # Additional temporal features
#         if 'Accident_Date' in df_eng.columns and 'Claims_Date' in df_eng.columns:
#             df_eng['Claim_Reporting_Delay'] = (
#                 pd.to_datetime(df_eng['Claims_Date']) - pd.to_datetime(df_eng['Accident_Date'])
#             ).dt.days
#             df_eng['Claim_Reporting_Delay'] = df_eng['Claim_Reporting_Delay'].fillna(0)
#             print("✓ Created Claim_Reporting_Delay")
        
#         if 'Policy_Start_Date' in df_eng.columns and 'Accident_Date' in df_eng.columns:
#             df_eng['Policy_Tenure_at_Accident'] = (
#                 pd.to_datetime(df_eng['Accident_Date']) - pd.to_datetime(df_eng['Policy_Start_Date'])
#             ).dt.days
#             df_eng['Policy_Tenure_at_Accident'] = df_eng['Policy_Tenure_at_Accident'].fillna(0)
#             print("✓ Created Policy_Tenure_at_Accident")
        
#         # Store engineered feature names
#         self.feature_names = [
#             'Claim_to_VehicleCost_Ratio', 'Premium_to_Claim_Ratio', 'Vehicle_Age',
#             'High_Mileage_Flag', 'Claim_Severity_Score', 'Claim_Reporting_Delay',
#             'Policy_Tenure_at_Accident'
#         ]
        
#         print(f"Total engineered features created: {len(self.feature_names)}")
#         return df_eng
    
#     def calculate_kpis(self, df):
#         """Calculate five key performance indicators with fraud breakdown"""
#         print("\n=== KPI CALCULATIONS ===")
        
#         kpis = {}
        
#         # KPI 1: Fraud Detection Rate
#         fraud_count = (df['Fraud_Ind'] == 'Y').sum()
#         total_claims = len(df)
#         kpis['fraud_rate'] = fraud_count / total_claims
#         kpis['fraud_count'] = fraud_count
#         kpis['legitimate_count'] = total_claims - fraud_count
        
#         print(f"KPI 1 - Fraud Rate: {kpis['fraud_rate']:.2%}")
#         print(f"  - Fraudulent Claims: {fraud_count:,}")
#         print(f"  - Legitimate Claims: {total_claims - fraud_count:,}")
        
#         # KPI 2: Average Claim Amounts by Status
#         fraud_df = df[df['Fraud_Ind'] == 'Y']
#         legit_df = df[df['Fraud_Ind'] == 'N']
        
#         kpis['avg_fraud_claim'] = fraud_df['Total_Claim'].mean() if len(fraud_df) > 0 else 0
#         kpis['avg_legit_claim'] = legit_df['Total_Claim'].mean() if len(legit_df) > 0 else 0
#         kpis['claim_amount_ratio'] = kpis['avg_fraud_claim'] / kpis['avg_legit_claim'] if kpis['avg_legit_claim'] > 0 else 0
        
#         print(f"KPI 2 - Average Claims:")
#         print(f"  - Fraudulent: ${kpis['avg_fraud_claim']:,.2f}")
#         print(f"  - Legitimate: ${kpis['avg_legit_claim']:,.2f}")
#         print(f"  - Ratio: {kpis['claim_amount_ratio']:.2f}x")
        
#         # KPI 3: Temporal Patterns
#         if 'Accident_Date' in df.columns:
#             df['Accident_Month'] = pd.to_datetime(df['Accident_Date']).dt.month
#             df['Accident_Hour_Group'] = pd.cut(df['Accident_Hour'], 
#                                              bins=[0, 6, 12, 18, 24], 
#                                              labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            
#             kpis['fraud_by_month'] = fraud_df.groupby('Accident_Month').size().to_dict() if len(fraud_df) > 0 else {}
#             kpis['fraud_by_hour_group'] = fraud_df.groupby('Accident_Hour_Group').size().to_dict() if len(fraud_df) > 0 else {}
            
#             print(f"KPI 3 - Temporal Analysis:")
#             print(f"  - Peak fraud month: {max(kpis['fraud_by_month'], key=kpis['fraud_by_month'].get) if kpis['fraud_by_month'] else 'N/A'}")
        
#         # KPI 4: Geographic Distribution
#         if 'Policy_State' in df.columns:
#             kpis['fraud_by_state'] = fraud_df['Policy_State'].value_counts().to_dict() if len(fraud_df) > 0 else {}
#             kpis['state_fraud_rates'] = {}
            
#             for state in df['Policy_State'].unique():
#                 state_df = df[df['Policy_State'] == state]
#                 state_fraud = (state_df['Fraud_Ind'] == 'Y').sum()
#                 kpis['state_fraud_rates'][state] = state_fraud / len(state_df) if len(state_df) > 0 else 0
            
#             top_fraud_state = max(kpis['state_fraud_rates'], key=kpis['state_fraud_rates'].get) if kpis['state_fraud_rates'] else 'N/A'
#             print(f"KPI 4 - Geographic Analysis:")
#             print(f"  - Highest fraud rate state: {top_fraud_state}")
        
#         # KPI 5: Vehicle and Claim Characteristics
#         if 'Vehicle_Age' in df.columns:
#             kpis['avg_fraud_vehicle_age'] = fraud_df['Vehicle_Age'].mean() if len(fraud_df) > 0 else 0
#             kpis['avg_legit_vehicle_age'] = legit_df['Vehicle_Age'].mean() if len(legit_df) > 0 else 0
            
#             print(f"KPI 5 - Vehicle Analysis:")
#             print(f"  - Avg fraud vehicle age: {kpis['avg_fraud_vehicle_age']:.1f} years")
#             print(f"  - Avg legit vehicle age: {kpis['avg_legit_vehicle_age']:.1f} years")
        
#         # High-value claim analysis
#         high_value_threshold = df['Total_Claim'].quantile(0.9)
#         high_value_claims = df[df['Total_Claim'] > high_value_threshold]
#         kpis['high_value_fraud_rate'] = (high_value_claims['Fraud_Ind'] == 'Y').sum() / len(high_value_claims) if len(high_value_claims) > 0 else 0
        
#         print(f"  - High-value claim fraud rate: {kpis['high_value_fraud_rate']:.2%}")
        
#         return kpis
    
#     def create_risk_segments(self, df):
#         """Create risk segments based on engineered features"""
#         print("\n=== RISK SEGMENTATION ===")
        
#         df_risk = df.copy()
        
#         # Create risk score based on multiple factors
#         risk_factors = []
        
#         # Factor 1: High claim to vehicle cost ratio
#         if 'Claim_to_VehicleCost_Ratio' in df_risk.columns:
#             high_ratio_threshold = df_risk['Claim_to_VehicleCost_Ratio'].quantile(0.8)
#             risk_factors.append(df_risk['Claim_to_VehicleCost_Ratio'] > high_ratio_threshold)
        
#         # Factor 2: Low premium to claim ratio
#         if 'Premium_to_Claim_Ratio' in df_risk.columns:
#             low_ratio_threshold = df_risk['Premium_to_Claim_Ratio'].quantile(0.2)
#             risk_factors.append(df_risk['Premium_to_Claim_Ratio'] < low_ratio_threshold)
        
#         # Factor 3: Old vehicle with high claim
#         if 'Vehicle_Age' in df_risk.columns:
#             old_vehicle_threshold = df_risk['Vehicle_Age'].quantile(0.8)
#             high_claim_threshold = df_risk['Total_Claim'].quantile(0.8)
#             risk_factors.append((df_risk['Vehicle_Age'] > old_vehicle_threshold) & 
#                               (df_risk['Total_Claim'] > high_claim_threshold))
        
#         # Factor 4: Late reporting
#         if 'Claim_Reporting_Delay' in df_risk.columns:
#             late_threshold = df_risk['Claim_Reporting_Delay'].quantile(0.8)
#             risk_factors.append(df_risk['Claim_Reporting_Delay'] > late_threshold)
        
#         # Factor 5: Early policy claim
#         if 'Policy_Tenure_at_Accident' in df_risk.columns:
#             early_threshold = 30  # Claims within 30 days of policy start
#             risk_factors.append(df_risk['Policy_Tenure_at_Accident'] < early_threshold)
        
#         # Calculate risk score (0-5 based on number of risk factors)
#         if risk_factors:
#             df_risk['Risk_Score'] = sum(risk_factors)
            
#             # Create risk segments
#             df_risk['Risk_Segment'] = pd.cut(
#                 df_risk['Risk_Score'], 
#                 bins=[-1, 0, 1, 2, 5], 
#                 labels=['Low', 'Medium', 'High', 'Critical']
#             )
            
#             print("Risk segments created:")
#             print(df_risk['Risk_Segment'].value_counts())
        
#         return df_risk

# def main():
#     """Main feature engineering pipeline"""
#     print("Starting feature engineering pipeline...")
    
#     # Load preprocessed data
#     try:
#         df = pd.read_csv('data/processed/clean_auto_insurance.csv')
#         print(f"Loaded preprocessed data: {len(df)} records")
#     except FileNotFoundError:
#         print("Preprocessed data not found. Please run data preprocessing first.")
#         return
    
#     # Initialize feature engineer
#     engineer = FeatureEngineer()
    
#     # Create engineered features
#     df_engineered = engineer.create_engineered_features(df)
    
#     # Calculate KPIs
#     kpis = engineer.calculate_kpis(df_engineered)
    
#     # Create risk segments
#     df_final = engineer.create_risk_segments(df_engineered)
    
#     # Save results
#     os.makedirs('data/processed', exist_ok=True)
#     df_final.to_csv('data/processed/engineered_auto_insurance.csv', index=False)
    
#     # Save KPIs
#     import json
#     with open('data/processed/kpis.json', 'w') as f:
#         # Convert numpy types to native Python types for JSON serialization
#         kpis_serializable = {}
#         for key, value in kpis.items():
#             if isinstance(value, (np.integer, np.floating)):
#                 kpis_serializable[key] = value.item()
#             elif isinstance(value, dict):
#                 kpis_serializable[key] = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) 
#                                         for k, v in value.items()}
#             else:
#                 kpis_serializable[key] = value
        
#         json.dump(kpis_serializable, f, indent=2)
    
#     print(f"\nFeature engineering completed!")
#     print(f"Final dataset shape: {df_final.shape}")
#     print(f"Engineered features: {engineer.feature_names}")
#     print("Results saved to 'data/processed/engineered_auto_insurance.csv'")
#     print("KPIs saved to 'data/processed/kpis.json'")

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_engineered_features(self, df):
        """Create seven new engineered features"""
        print("\n=== FEATURE ENGINEERING ===")
        
        df_eng = df.copy()
        
        # Feature 1: Claim to Vehicle Cost Ratio
        df_eng['Claim_to_VehicleCost_Ratio'] = df_eng['Total_Claim'] / (df_eng['Vehicle_Cost'] + 1)
        print("✓ Created Claim_to_VehicleCost_Ratio")
        
        # Feature 2: Premium to Claim Ratio
        df_eng['Premium_to_Claim_Ratio'] = df_eng['Policy_Premium'] / (df_eng['Total_Claim'] + 1)
        print("✓ Created Premium_to_Claim_Ratio")
        
        # Feature 3: Vehicle Age
        current_year = datetime.now().year
        df_eng['Vehicle_Age'] = current_year - df_eng['Auto_Year']
        df_eng['Vehicle_Age'] = df_eng['Vehicle_Age'].clip(lower=0)
        print("✓ Created Vehicle_Age")
        
        # Feature 4: High Mileage Flag
        median_mileage = df_eng['Annual_Mileage'].median()
        df_eng['High_Mileage_Flag'] = (df_eng['Annual_Mileage'] > median_mileage).astype(int)
        print(f"✓ Created High_Mileage_Flag (threshold: {median_mileage:,.0f} miles)")
        
        # Feature 5: Claim Severity Score
        # Weighted composite score: Injury (50%) + Property (30%) + Vehicle (20%)
        max_injury = df_eng['Injury_Claim'].max()
        max_property = df_eng['Property_Claim'].max()
        max_vehicle = df_eng['Vehicle_Claim'].max()
        
        if max_injury > 0 and max_property > 0 and max_vehicle > 0:
            df_eng['Claim_Severity_Score'] = (
                (df_eng['Injury_Claim'] / max_injury) * 0.5 + 
                (df_eng['Property_Claim'] / max_property) * 0.3 + 
                (df_eng['Vehicle_Claim'] / max_vehicle) * 0.2
            )
        else:
            df_eng['Claim_Severity_Score'] = 0
        
        print("✓ Created Claim_Severity_Score")
        
        # Convert date columns with errors='coerce' to handle invalid dates gracefully
        # This is important before creating temporal features
        for date_col in ['Accident_Date', 'Claims_Date', 'Policy_Start_Date']:
            if date_col in df_eng.columns:
                df_eng[date_col] = pd.to_datetime(df_eng[date_col], errors='coerce')
        
        # Feature 6: Claim Reporting Delay (days between claim and accident)
        if 'Accident_Date' in df_eng.columns and 'Claims_Date' in df_eng.columns:
            df_eng['Claim_Reporting_Delay'] = (df_eng['Claims_Date'] - df_eng['Accident_Date']).dt.days
            df_eng['Claim_Reporting_Delay'] = df_eng['Claim_Reporting_Delay'].fillna(0)
            print("✓ Created Claim_Reporting_Delay")
        else:
            df_eng['Claim_Reporting_Delay'] = 0
        
        # Feature 7: Policy Tenure at Accident (days since policy start to accident date)
        if 'Policy_Start_Date' in df_eng.columns and 'Accident_Date' in df_eng.columns:
            df_eng['Policy_Tenure_at_Accident'] = (df_eng['Accident_Date'] - df_eng['Policy_Start_Date']).dt.days
            df_eng['Policy_Tenure_at_Accident'] = df_eng['Policy_Tenure_at_Accident'].fillna(0)
            print("✓ Created Policy_Tenure_at_Accident")
        else:
            df_eng['Policy_Tenure_at_Accident'] = 0
        
        self.feature_names = [
            'Claim_to_VehicleCost_Ratio', 'Premium_to_Claim_Ratio', 'Vehicle_Age',
            'High_Mileage_Flag', 'Claim_Severity_Score', 'Claim_Reporting_Delay',
            'Policy_Tenure_at_Accident'
        ]
        
        print(f"Total engineered features created: {len(self.feature_names)}")
        return df_eng
    
    def calculate_kpis(self, df):
        """Calculate five key performance indicators with fraud breakdown"""
        print("\n=== KPI CALCULATIONS ===")
        
        kpis = {}
        
        # KPI 1: Fraud Detection Rate
        fraud_count = (df['Fraud_Ind'] == 'Y').sum()
        total_claims = len(df)
        kpis['fraud_rate'] = fraud_count / total_claims
        kpis['fraud_count'] = fraud_count
        kpis['legitimate_count'] = total_claims - fraud_count
        
        print(f"KPI 1 - Fraud Rate: {kpis['fraud_rate']:.2%}")
        print(f"  - Fraudulent Claims: {fraud_count:,}")
        print(f"  - Legitimate Claims: {total_claims - fraud_count:,}")
        
        # KPI 2: Average Claim Amounts by Status
        fraud_df = df[df['Fraud_Ind'] == 'Y']
        legit_df = df[df['Fraud_Ind'] == 'N']
        
        kpis['avg_fraud_claim'] = fraud_df['Total_Claim'].mean() if len(fraud_df) > 0 else 0
        kpis['avg_legit_claim'] = legit_df['Total_Claim'].mean() if len(legit_df) > 0 else 0
        kpis['claim_amount_ratio'] = (kpis['avg_fraud_claim'] / kpis['avg_legit_claim']) if kpis['avg_legit_claim'] > 0 else 0
        
        print(f"KPI 2 - Average Claims:")
        print(f"  - Fraudulent: ${kpis['avg_fraud_claim']:,.2f}")
        print(f"  - Legitimate: ${kpis['avg_legit_claim']:,.2f}")
        print(f"  - Ratio: {kpis['claim_amount_ratio']:.2f}x")
        
        # Create 'Accident_Month' column before grouping
        if 'Accident_Date' in df.columns:
            df['Accident_Month'] = pd.to_datetime(df['Accident_Date'], errors='coerce').dt.month
        else:
            df['Accident_Month'] = np.nan
        
        if 'Accident_Hour' in df.columns:
            # Create time groups (e.g., Night, Morning, Afternoon, Evening)
            bins = [0, 6, 12, 18, 24]
            labels = ['Night', 'Morning', 'Afternoon', 'Evening']
            df['Accident_Hour_Group'] = pd.cut(df['Accident_Hour'], bins=bins, labels=labels, right=False)
        else:
            df['Accident_Hour_Group'] = np.nan
        
        # Ensure these columns are in fraud_df
        fraud_df = df[df['Fraud_Ind'] == 'Y']
        
        # KPI 3: Temporal Patterns
        kpis['fraud_by_month'] = fraud_df.groupby('Accident_Month').size().to_dict() if len(fraud_df) > 0 else {}
        kpis['fraud_by_hour_group'] = fraud_df.groupby('Accident_Hour_Group').size().to_dict() if len(fraud_df) > 0 else {}
        
        print(f"KPI 3 - Temporal Analysis:")
        if kpis['fraud_by_month']:
            peak_month = max(kpis['fraud_by_month'], key=kpis['fraud_by_month'].get)
            print(f"  - Peak fraud month: {peak_month}")
        else:
            print("  - No Accident_Month data available")
        
        # KPI 4: Geographic Distribution
        if 'Policy_State' in df.columns:
            kpis['fraud_by_state'] = fraud_df['Policy_State'].value_counts().to_dict() if len(fraud_df) > 0 else {}
            kpis['state_fraud_rates'] = {}
            
            for state in df['Policy_State'].unique():
                state_df = df[df['Policy_State'] == state]
                if len(state_df) > 0:
                    state_fraud = (state_df['Fraud_Ind'] == 'Y').sum()
                    kpis['state_fraud_rates'][state] = state_fraud / len(state_df)
                else:
                    kpis['state_fraud_rates'][state] = 0
            
            if kpis['state_fraud_rates']:
                top_fraud_state = max(kpis['state_fraud_rates'], key=kpis['state_fraud_rates'].get)
                print(f"KPI 4 - Geographic Analysis:")
                print(f"  - Highest fraud rate state: {top_fraud_state}")
            else:
                print("  - No Policy_State data available for fraud rates")
        else:
            kpis['fraud_by_state'] = {}
            kpis['state_fraud_rates'] = {}
            print("  - Policy_State column not found for geographic analysis")
        
        # KPI 5: Vehicle and Claim Characteristics
        if 'Vehicle_Age' in df.columns:
            kpis['avg_fraud_vehicle_age'] = fraud_df['Vehicle_Age'].mean() if len(fraud_df) > 0 else 0
            kpis['avg_legit_vehicle_age'] = legit_df['Vehicle_Age'].mean() if len(legit_df) > 0 else 0
            
            print(f"KPI 5 - Vehicle Analysis:")
            print(f"  - Avg fraud vehicle age: {kpis['avg_fraud_vehicle_age']:.1f} years")
            print(f"  - Avg legit vehicle age: {kpis['avg_legit_vehicle_age']:.1f} years")
        else:
            kpis['avg_fraud_vehicle_age'] = 0
            kpis['avg_legit_vehicle_age'] = 0
        
        # High-value claim analysis
        high_value_threshold = df['Total_Claim'].quantile(0.9)
        high_value_claims = df[df['Total_Claim'] > high_value_threshold]
        if len(high_value_claims) > 0:
            kpis['high_value_fraud_rate'] = (high_value_claims['Fraud_Ind'] == 'Y').sum() / len(high_value_claims)
        else:
            kpis['high_value_fraud_rate'] = 0
        
        print(f"  - High-value claim fraud rate: {kpis['high_value_fraud_rate']:.2%}")
        
        return kpis
    
    def create_risk_segments(self, df):
        """Create risk segments based on engineered features"""
        print("\n=== RISK SEGMENTATION ===")
        
        df_risk = df.copy()
        
        # Create risk factors list
        risk_factors = []
        
        # Factor 1: High claim to vehicle cost ratio
        if 'Claim_to_VehicleCost_Ratio' in df_risk.columns:
            high_ratio_threshold = df_risk['Claim_to_VehicleCost_Ratio'].quantile(0.8)
            risk_factors.append(df_risk['Claim_to_VehicleCost_Ratio'] > high_ratio_threshold)
        
        # Factor 2: Low premium to claim ratio
        if 'Premium_to_Claim_Ratio' in df_risk.columns:
            low_ratio_threshold = df_risk['Premium_to_Claim_Ratio'].quantile(0.2)
            risk_factors.append(df_risk['Premium_to_Claim_Ratio'] < low_ratio_threshold)
        
        # Factor 3: Old vehicle & high claim
        if 'Vehicle_Age' in df_risk.columns:
            old_vehicle_threshold = df_risk['Vehicle_Age'].quantile(0.8)
            high_claim_threshold = df_risk['Total_Claim'].quantile(0.8)
            risk_factors.append((df_risk['Vehicle_Age'] > old_vehicle_threshold) & 
                              (df_risk['Total_Claim'] > high_claim_threshold))
        
        # Factor 4: Late reporting
        if 'Claim_Reporting_Delay' in df_risk.columns:
            late_threshold = df_risk['Claim_Reporting_Delay'].quantile(0.8)
            risk_factors.append(df_risk['Claim_Reporting_Delay'] > late_threshold)
        
        # Factor 5: Early policy claim
        if 'Policy_Tenure_at_Accident' in df_risk.columns:
            early_threshold = 30  # days
            risk_factors.append(df_risk['Policy_Tenure_at_Accident'] < early_threshold)
        
        if risk_factors:
            df_risk['Risk_Score'] = sum(risk_factors).astype(int)
            
            # Define risk segments
            df_risk['Risk_Segment'] = pd.cut(
                df_risk['Risk_Score'],
                bins=[-1, 0, 1, 2, 5],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            print("Risk segments created:")
            print(df_risk['Risk_Segment'].value_counts())
        
        return df_risk

def main():
    """Main feature engineering pipeline"""
    print("Starting feature engineering pipeline...")
    
    try:
        df = pd.read_csv('data/processed/clean_auto_insurance.csv')
        print(f"Loaded preprocessed data: {len(df)} records")
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data preprocessing first.")
        return
    
    engineer = FeatureEngineer()
    
    df_engineered = engineer.create_engineered_features(df)
    
    kpis = engineer.calculate_kpis(df_engineered)
    
    df_final = engineer.create_risk_segments(df_engineered)
    
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/engineered_auto_insurance.csv', index=False)
    
    # Save KPIs as JSON
    import json
    with open('data/processed/kpis.json', 'w') as f:
        # Convert numpy types for serialization
        def convert_np(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(i) for i in obj]
            else:
                return obj
        json.dump(convert_np(kpis), f, indent=2)
    
    print("\nFeature engineering completed!")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Engineered features: {engineer.feature_names}")
    print("Results saved to 'data/processed/engineered_auto_insurance.csv'")
    print("KPIs saved to 'data/processed/kpis.json'")

if __name__ == "__main__":
    main()
