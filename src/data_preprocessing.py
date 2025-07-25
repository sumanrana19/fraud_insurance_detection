import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def load_and_merge_data(self, data_path='data/raw/'):
        """Load and merge all CSV part files"""
        print("Loading data files...")
        
        # Get all CSV files in the raw data directory
        csv_files = glob.glob(os.path.join(data_path, "mergedataA_part*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        dataframes = []
        for file in sorted(csv_files):
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
                print(f"Loaded {file}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dataframes:
            raise ValueError("No data files were successfully loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total combined records: {len(combined_df)}")
        
        return combined_df
    
    def analyze_duplicates(self, df):
        """Analyze and identify duplicate records"""
        print("\n=== DUPLICATE ANALYSIS ===")
        
        # Check for duplicate Claim_IDs
        duplicate_claim_ids = df[df.duplicated(subset=['Claim_ID'], keep=False)]
        
        print(f"Total records: {len(df)}")
        print(f"Duplicate Claim_IDs found: {len(duplicate_claim_ids)}")
        
        if len(duplicate_claim_ids) > 0:
            print("\nDuplicate Claim_ID details:")
            duplicate_summary = duplicate_claim_ids.groupby('Claim_ID').size().sort_values(ascending=False)
            print(duplicate_summary.head(10))
            
            # Save duplicate analysis
            os.makedirs('data/processed', exist_ok=True)
            duplicate_claim_ids.to_csv('data/processed/duplicate_claim_ids.csv', index=False)
            print("Duplicate records saved to 'data/processed/duplicate_claim_ids.csv'")
        else:
            print("✅ No duplicate Claim_IDs found - excellent data quality!")
        
        # Check for completely identical rows
        complete_duplicates = df[df.duplicated(keep=False)]
        print(f"Complete duplicate rows: {len(complete_duplicates)}")
        
        return duplicate_claim_ids
    
    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies"""
        print("\n=== MISSING VALUE ANALYSIS ===")
        
        missing_info = df.isnull().sum()
        missing_percent = (missing_info / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_info.index,
            'Missing_Count': missing_info.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
        
        print("Missing values by column:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Handle missing values by data type
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Numerical columns - use median
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} with median: {median_val}")
                else:
                    # Categorical columns - use mode
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
                        print(f"Filled {col} with mode: {mode_val[0]}")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"Filled {col} with 'Unknown'")
        
        return df
    
    def remove_outliers(self, df, method='iqr', factor=1.5):
        """Remove outliers using IQR method"""
        print("\n=== OUTLIER REMOVAL ===")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['Claim_ID']]
        
        outliers_removed = 0
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_removed += outliers_count
            
            # Clip outliers instead of removing
            df[col] = df[col].clip(lower_bound, upper_bound)
            
            if outliers_count > 0:
                print(f"{col}: {outliers_count} outliers clipped to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        print(f"Total outliers handled: {outliers_removed}")
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables"""
        print("\n=== CATEGORICAL ENCODING ===")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['Claim_ID', 'Vehicle_Registration']]
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            print(f"Encoded {col}: {len(self.label_encoders[col].classes_)} unique values")
        
        return df
    
    def preprocess_data(self, data_path='data/raw/'):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_and_merge_data(data_path)
        
        # Analyze duplicates
        duplicates = self.analyze_duplicates(df)
        
        # Remove duplicates (keep first occurrence)
        df_clean = df.drop_duplicates(subset=['Claim_ID'], keep='first')
        print(f"Records after deduplication: {len(df_clean)}")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Remove outliers
        df_clean = self.remove_outliers(df_clean)
        
        # Convert date columns
        date_columns = ['Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date', 
                       'Accident_Date', 'DL_Expiry_Date', 'Claims_Date']
        
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Save preprocessed data
        os.makedirs('data/processed', exist_ok=True)
        df_clean.to_csv('data/processed/clean_auto_insurance.csv', index=False)
        print(f"Preprocessed data saved: {len(df_clean)} records")
        
        return df_clean

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess_data()
    print("Data preprocessing completed successfully!")
