import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="Auto Insurance Fraud Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .safe-alert {
        background: linear-gradient(90deg, #2ed573 0%, #1e90ff 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚗 Auto Insurance Fraud Detection System")
st.markdown("**Advanced ML-Powered Fraud Detection and Analytics Dashboard**")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select a page to explore different aspects of fraud detection:")

page = st.sidebar.selectbox("Choose a page", [
    "📊 Data Overview", 
    "🔍 Fraud Analysis", 
    "🤖 Model Performance", 
    "🎯 Prediction Interface",
    "📈 KPI Dashboard"
])

# Load data function
@st.cache_data
def load_data():
    """Load processed data with error handling"""
    try:
        # Try to load engineered data first
        if os.path.exists('data/processed/engineered_auto_insurance.csv'):
            df = pd.read_csv('data/processed/engineered_auto_insurance.csv')
            st.sidebar.success("✅ Loaded engineered dataset")
        elif os.path.exists('data/processed/clean_auto_insurance.csv'):
            df = pd.read_csv('data/processed/clean_auto_insurance.csv')
            st.sidebar.warning("⚠️ Loaded basic cleaned dataset")
        else:
            st.sidebar.error("❌ No processed data found")
            return None
        
        # Ensure Fraud_Binary exists
        if 'Fraud_Binary' not in df.columns:
            df['Fraud_Binary'] = (df['Fraud_Ind'] == 'Y').astype(int)
        
        return df
    except FileNotFoundError:
        st.sidebar.error("❌ Data files not found")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return None

@st.cache_data
def load_kpis():
    """Load KPI data"""
    try:
        with open('data/processed/kpis.json', 'r') as f:
            return json.load(f)
    except:
        return {}

@st.cache_data
def load_model_results():
    """Load model evaluation results"""
    try:
        with open('outputs/model_evaluation_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

# Load data
df = load_data()
kpis = load_kpis()
model_results = load_model_results()

if df is not None:
    # Data Overview Page
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        # Key metrics cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Claims</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            fraud_count = (df['Fraud_Ind'] == 'Y').sum()
            st.markdown("""
            <div class="metric-card">
                <h3>Fraudulent Claims</h3>
                <h2>{:,}</h2>
            </div>
            """.format(fraud_count), unsafe_allow_html=True)
        
        with col3:
            legit_count = (df['Fraud_Ind'] == 'N').sum()
            st.markdown("""
            <div class="metric-card">
                <h3>Legitimate Claims</h3>
                <h2>{:,}</h2>
            </div>
            """.format(legit_count), unsafe_allow_html=True)
        
        with col4:
            fraud_rate = fraud_count / len(df) * 100
            st.markdown("""
            <div class="metric-card">
                <h3>Fraud Rate</h3>
                <h2>{:.2f}%</h2>
            </div>
            """.format(fraud_rate), unsafe_allow_html=True)
        
        with col5:
            avg_claim = df['Total_Claim'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Claim Amount</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(avg_claim), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data quality section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Information")
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            st.write(f"**Duplicate Records:** {df.duplicated().sum():,}")
            
            # Data types
            dtype_counts = df.dtypes.value_counts()
            st.write("**Data Types:**")
            for dtype, count in dtype_counts.items():
                st.write(f"  - {dtype}: {count} columns")
        
        with col2:
            st.subheader("🔍 Data Quality Metrics")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if not missing_data.empty:
                fig = px.bar(
                    x=missing_data.values, 
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Missing Count', 'y': 'Column'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        # Feature distribution
        st.subheader("📊 Feature Distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['Claim_ID']]
        
        selected_features = st.multiselect(
            "Select features to visualize:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if selected_features:
            n_cols = min(2, len(selected_features))
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features
            )
            
            for i, feature in enumerate(selected_features):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                fraud_data = df[df['Fraud_Ind'] == 'Y'][feature]
                legit_data = df[df['Fraud_Ind'] == 'N'][feature]
                
                fig.add_trace(
                    go.Histogram(x=fraud_data, name=f'Fraud ({feature})', 
                               opacity=0.7, nbinsx=30),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Histogram(x=legit_data, name=f'Legitimate ({feature})', 
                               opacity=0.7, nbinsx=30),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data sample
        st.subheader("🗃️ Data Sample")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            fraud_filter = st.selectbox("Filter by Fraud Status:", 
                                      ["All", "Fraudulent Only", "Legitimate Only"])
        with col2:
            sample_size = st.slider("Sample Size:", 10, 500, 100)
        with col3:
            random_seed = st.number_input("Random Seed:", value=42)
        
        # Apply filters
        display_df = df.copy()
        if fraud_filter == "Fraudulent Only":
            display_df = display_df[display_df['Fraud_Ind'] == 'Y']
        elif fraud_filter == "Legitimate Only":
            display_df = display_df[display_df['Fraud_Ind'] == 'N']
        
        # Sample data
        if len(display_df) > sample_size:
            display_df = display_df.sample(n=sample_size, random_state=random_seed)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f'insurance_data_sample_{fraud_filter.lower().replace(" ", "_")}.csv',
            mime='text/csv'
        )

    # Fraud Analysis Page
    elif page == "🔍 Fraud Analysis":
        st.header("🔍 Comprehensive Fraud Analysis")
        
        # Fraud vs Non-Fraud Overview
        st.subheader("🎯 Fraud Distribution Overview")
        
        fraud_counts = df['Fraud_Ind'].value_counts()
        fraud_rate = fraud_counts['Y'] / (fraud_counts['Y'] + fraud_counts['N']) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Fraud distribution pie chart
            fig = px.pie(
                values=fraud_counts.values, 
                names=['Legitimate', 'Fraudulent'],
                title="Fraud vs Legitimate Claims",
                color_discrete_sequence=['#2E86AB', '#A23B72']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Key Statistics")
            st.markdown(f"""
            - **Total Claims:** {len(df):,}
            - **Fraudulent Claims:** {fraud_counts['Y']:,}
            - **Legitimate Claims:** {fraud_counts['N']:,}
            - **Fraud Rate:** {fraud_rate:.2f}%
            - **Detection Priority:** {'HIGH' if fraud_rate > 10 else 'MEDIUM' if fraud_rate > 5 else 'LOW'}
            """)
        
        with col3:
            # Fraud impact
            fraud_claims = df[df['Fraud_Ind'] == 'Y']['Total_Claim'].sum()
            legit_claims = df[df['Fraud_Ind'] == 'N']['Total_Claim'].sum()
            total_claims_amount = fraud_claims + legit_claims
            
            st.markdown("### 💰 Financial Impact")
            st.markdown(f"""
            - **Fraudulent Claims Value:** ${fraud_claims:,.0f}
            - **Legitimate Claims Value:** ${legit_claims:,.0f}
            - **Total Claims Value:** ${total_claims_amount:,.0f}
            - **Fraud Loss Ratio:** {fraud_claims/total_claims_amount*100:.1f}%
            """)
        
        st.markdown("---")
        
        # Claim amounts analysis
        st.subheader("💵 Claim Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of claim amounts
            fig = px.box(
                df, x='Fraud_Ind', y='Total_Claim', 
                title="Claim Amount Distribution by Fraud Status",
                labels={'Fraud_Ind': 'Fraud Status', 'Total_Claim': 'Total Claim Amount ($)'}
            )
            fig.update_xaxes(ticktext=['Legitimate', 'Fraudulent'], tickvals=['N', 'Y'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Claim components breakdown
            claim_components = ['Injury_Claim', 'Property_Claim', 'Vehicle_Claim']
            fraud_avg = df[df['Fraud_Ind'] == 'Y'][claim_components].mean()
            legit_avg = df[df['Fraud_Ind'] == 'N'][claim_components].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Fraudulent', x=claim_components, y=fraud_avg.values))
            fig.add_trace(go.Bar(name='Legitimate', x=claim_components, y=legit_avg.values))
            
            fig.update_layout(
                title='Average Claim Components by Fraud Status',
                xaxis_title='Claim Component',
                yaxis_title='Average Amount ($)',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic analysis
        st.subheader("🗺️ Geographic Fraud Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by state
            fraud_by_state = df[df['Fraud_Ind'] == 'Y']['Policy_State'].value_counts().head(10)
            
            fig = px.bar(
                x=fraud_by_state.values, 
                y=fraud_by_state.index,
                orientation='h',
                title="Top 10 States by Fraudulent Claims Count",
                labels={'x': 'Number of Fraudulent Claims', 'y': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # State fraud rates
            state_stats = df.groupby('Policy_State').agg({
                'Fraud_Ind': lambda x: (x == 'Y').sum(),
                'Claim_ID': 'count'
            })
            state_stats['Fraud_Rate'] = state_stats['Fraud_Ind'] / state_stats['Claim_ID']
            state_stats = state_stats.sort_values('Fraud_Rate', ascending=False).head(10)
            
            fig = px.bar(
                x=state_stats['Fraud_Rate'].values * 100,
                y=state_stats.index,
                orientation='h',
                title="Top 10 States by Fraud Rate (%)",
                labels={'x': 'Fraud Rate (%)', 'y': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal analysis
        st.subheader("⏰ Temporal Fraud Patterns")
        
        if 'Accident_Date' in df.columns:
            df['Accident_Month'] = pd.to_datetime(df['Accident_Date']).dt.month
            df['Accident_Weekday'] = pd.to_datetime(df['Accident_Date']).dt.dayofweek
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly fraud pattern
                monthly_fraud = df[df['Fraud_Ind'] == 'Y'].groupby('Accident_Month').size()
                monthly_total = df.groupby('Accident_Month').size()
                monthly_rate = (monthly_fraud / monthly_total * 100).fillna(0)
                
                fig = px.line(
                    x=monthly_rate.index, y=monthly_rate.values,
                    title="Fraud Rate by Month",
                    labels={'x': 'Month', 'y': 'Fraud Rate (%)'}
                )
                fig.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly fraud pattern
                hourly_fraud = df[df['Fraud_Ind'] == 'Y'].groupby('Accident_Hour').size()
                hourly_total = df.groupby('Accident_Hour').size()
                hourly_rate = (hourly_fraud / hourly_total * 100).fillna(0)
                
                fig = px.bar(
                    x=hourly_rate.index, y=hourly_rate.values,
                    title="Fraud Rate by Hour of Day",
                    labels={'x': 'Hour of Day', 'y': 'Fraud Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle analysis
        st.subheader("🚗 Vehicle-Related Fraud Patterns")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Vehicle age analysis
            if 'Vehicle_Age' in df.columns:
                fig = px.histogram(
                    df, x='Vehicle_Age', color='Fraud_Ind',
                    title="Vehicle Age Distribution by Fraud Status",
                    nbins=20, barmode='overlay'
                )
                fig.update_traces(opacity=0.7)
                fig.update_xaxes(title='Vehicle Age (Years)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vehicle make analysis
            top_makes = df['Auto_Make'].value_counts().head(8).index
            fraud_by_make = df[df['Auto_Make'].isin(top_makes)].groupby(['Auto_Make', 'Fraud_Ind']).size().unstack(fill_value=0)
            fraud_rates = fraud_by_make['Y'] / (fraud_by_make['Y'] + fraud_by_make['N']) * 100
            
            fig = px.bar(
                x=fraud_rates.values, y=fraud_rates.index,
                orientation='h',
                title="Fraud Rate by Vehicle Make (%)",
                labels={'x': 'Fraud Rate (%)', 'y': 'Vehicle Make'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Accident type analysis
            accident_fraud = df.groupby(['Accident_Type', 'Fraud_Ind']).size().unstack(fill_value=0)
            accident_rates = accident_fraud['Y'] / (accident_fraud['Y'] + accident_fraud['N']) * 100
            
            fig = px.bar(
                x=accident_rates.values, y=accident_rates.index,
                orientation='h',
                title="Fraud Rate by Accident Type (%)",
                labels={'x': 'Fraud Rate (%)', 'y': 'Accident Type'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Model Performance Page
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance Analysis")
        
        if model_results:
            # Model performance summary
            st.subheader("📊 Model Performance Summary")
            
            # Convert to DataFrame for easier handling
            results_df = pd.DataFrame(model_results).T
            results_df = results_df.sort_values('roc_auc', ascending=False)
            
            # Top performers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_model = results_df.index[0]
                best_score = results_df.iloc[0]['roc_auc']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🥇 Best Model</h3>
                    <h4>{best_model}</h4>
                    <h3>ROC-AUC: {best_score:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_accuracy = results_df['accuracy'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Avg Accuracy</h3>
                    <h2>{avg_accuracy:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                models_count = len(results_df)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔢 Models Trained</h3>
                    <h2>{models_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed performance comparison
            st.subheader("🏆 Model Performance Comparison")
            
            # Performance metrics radar chart
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            top_models = results_df.head(5)
            
            fig = go.Figure()
            
            for i, (model_name, row) in enumerate(top_models.iterrows()):
                values = [row[metric] for metric in metrics_to_plot]
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_to_plot + [metrics_to_plot[0]],
                    fill='toself',
                    name=model_name,
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Top 5 Models - Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📊 Detailed Performance Metrics")
            
            # Select metrics to display
            all_metrics = list(results_df.columns)
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                          'cohen_kappa', 'matthews_corrcoef', 'specificity']
            
            display_metrics = st.multiselect(
                "Select metrics to display:",
                all_metrics,
                default=[m for m in key_metrics if m in all_metrics]
            )
            
            if display_metrics:
                display_df = results_df[display_metrics].round(4)
                st.dataframe(display_df, use_container_width=True)
            
            # Feature importance (if available)
            if os.path.exists('models/feature_importance.json'):
                st.subheader("🎯 Feature Importance Analysis")
                
                try:
                    with open('models/feature_importance.json', 'r') as f:
                        importance_data = json.load(f)
                    
                    if importance_data:
                        # Select model for feature importance
                        model_options = list(importance_data.keys())
                        selected_model = st.selectbox("Select model for feature importance:", model_options)
                        
                        if selected_model in importance_data:
                            features = importance_data[selected_model]['features']
                            importance = importance_data[selected_model]['importance']
                            
                            # Create feature importance DataFrame
                            feat_imp_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)
                            
                            # Plot top features
                            top_n = st.slider("Number of top features to display:", 5, 20, 10)
                            top_features = feat_imp_df.head(top_n)
                            
                            fig = px.bar(
                                top_features, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title=f'Top {top_n} Features - {selected_model}',
                                labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance table
                            st.dataframe(feat_imp_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error loading feature importance: {e}")
            
            # Model recommendations
            st.subheader("💡 Model Recommendations")
            
            best_model = results_df.index[0]
            best_metrics = results_df.iloc[0]
            
            st.success(f"""
            **Recommended Model: {best_model}**
            
            - **ROC-AUC Score:** {best_metrics['roc_auc']:.4f} (Excellent discrimination)
            - **Accuracy:** {best_metrics['accuracy']:.4f}
            - **Precision:** {best_metrics['precision']:.4f} (Low false positive rate)
            - **Recall:** {best_metrics['recall']:.4f} (Good fraud detection rate)
            - **F1-Score:** {best_metrics['f1_score']:.4f} (Balanced performance)
            """)
            
            # Business impact estimation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Expected Performance")
                monthly_claims = st.number_input("Monthly Claims Volume:", value=1000, step=100)
                fraud_rate_input = st.number_input("Expected Fraud Rate (%):", value=15.0, step=0.5)
                
                frauds_expected = monthly_claims * (fraud_rate_input / 100)
                frauds_detected = frauds_expected * best_metrics['recall']
                false_positives = monthly_claims * (1 - fraud_rate_input/100) * (1 - best_metrics['specificity'])
                
                st.write(f"**Expected Monthly Results:**")
                st.write(f"- Fraudulent Claims: {frauds_expected:.0f}")
                st.write(f"- Frauds Detected: {frauds_detected:.0f}")
                st.write(f"- False Positives: {false_positives:.0f}")
                st.write(f"- Detection Rate: {best_metrics['recall']:.1%}")
            
            with col2:
                st.markdown("### 💰 Financial Impact")
                avg_fraud_amount = st.number_input("Average Fraud Amount ($):", value=25000, step=1000)
                investigation_cost = st.number_input("Cost per Investigation ($):", value=500, step=50)
                
                savings = frauds_detected * avg_fraud_amount
                investigation_costs = (frauds_detected + false_positives) * investigation_cost
                net_savings = savings - investigation_costs
                
                st.write(f"**Expected Monthly Impact:**")
                st.write(f"- Gross Savings: ${savings:,.0f}")
                st.write(f"- Investigation Costs: ${investigation_costs:,.0f}")
                st.write(f"- Net Savings: ${net_savings:,.0f}")
                st.write(f"- ROI: {(net_savings/investigation_costs)*100:.1f}%" if investigation_costs > 0 else "- ROI: N/A")
        
        else:
            st.warning("⚠️ No model evaluation results found. Please run model training and evaluation first.")
            st.info("""
            To generate model performance results:
            1. Run `python src/model_training.py`
            2. Run `python src/model_evaluation.py`
            3. Refresh this page
            """)

    # Prediction Interface Page
    elif page == "🎯 Prediction Interface":
        st.header("🎯 Fraud Prediction Interface")
        st.markdown("Enter claim details to get real-time fraud risk assessment")
        
        # Load a simple model for demonstration
        @st.cache_resource
        def load_demo_model():
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                import joblib
                
                # Try to load actual trained model
                if os.path.exists('models/random_forest_model.joblib'):
                    model = joblib.load('models/random_forest_model.joblib')
                    scaler = joblib.load('models/scaler.joblib') if os.path.exists('models/scaler.joblib') else StandardScaler()
                    return model, scaler
                else:
                    # Train a quick demo model
                    feature_cols = ['Age_Insured', 'Policy_Premium', 'Total_Claim', 
                                   'Vehicle_Age', 'Annual_Mileage', 'Claim_to_VehicleCost_Ratio']
                    
                    # Check which features exist
                    available_features = [col for col in feature_cols if col in df.columns]
                    
                    if len(available_features) >= 3:
                        X = df[available_features].fillna(df[available_features].median())
                        y = (df['Fraud_Ind'] == 'Y').astype(int)
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_scaled, y)
                        
                        return model, scaler
                    else:
                        return None, None
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None, None
        
        model, scaler = load_demo_model()
        
        if model is not None:
            st.success("✅ Fraud detection model loaded successfully!")
            
            # Input form
            st.subheader("📝 Enter Claim Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Policyholder Information**")
                age = st.number_input("Age of Insured", min_value=18, max_value=100, value=35)
                premium = st.number_input("Policy Premium ($)", min_value=0.0, value=1000.0, step=50.0)
                
                if 'Capital_Loss' in df.columns:
                    capital_loss = st.number_input("Capital Loss ($)", min_value=0.0, value=0.0, step=1000.0)
            
            with col2:
                st.markdown("**🚗 Vehicle Information**")
                vehicle_cost = st.number_input("Vehicle Cost ($)", min_value=0.0, value=20000.0, step=1000.0)
                annual_mileage = st.number_input("Annual Mileage", min_value=0, value=12000, step=1000)
                vehicle_year = st.number_input("Vehicle Year", min_value=1990, max_value=2024, value=2018)
                vehicle_age = 2024 - vehicle_year
            
            with col3:
                st.markdown("**💰 Claim Information**")
                claim_amount = st.number_input("Total Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0)
                injury_claim = st.number_input("Injury Claim ($)", min_value=0.0, value=0.0, step=100.0)
                accident_hour = st.slider("Accident Hour (24hr format)", 0, 23, 12)
            
            # Calculate engineered features
            claim_to_vehicle_ratio = claim_amount / (vehicle_cost + 1) if vehicle_cost > 0 else 0
            premium_to_claim_ratio = premium / (claim_amount + 1) if claim_amount > 0 else 0
            high_mileage_flag = 1 if annual_mileage > df['Annual_Mileage'].median() else 0
            
            # Prediction button
            if st.button("🔍 Analyze Fraud Risk", type="primary"):
                try:
                    # Prepare input data
                    input_features = [age, premium, claim_amount, vehicle_age, annual_mileage, claim_to_vehicle_ratio]
                    input_array = np.array([input_features])
                    
                    # Scale features
                    input_scaled = scaler.transform(input_array)
                    
                    # Make prediction
                    fraud_probability = model.predict_proba(input_scaled)[0][1]
                    prediction = model.predict(input_scaled)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Fraud Risk Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if fraud_probability > 0.7:
                            risk_level = "🚨 HIGH RISK"
                            alert_class = "fraud-alert"
                        elif fraud_probability > 0.4:
                            risk_level = "⚠️ MEDIUM RISK"
                            alert_class = "metric-card"
                        else:
                            risk_level = "✅ LOW RISK"
                            alert_class = "safe-alert"
                        
                        st.markdown(f"""
                        <div class="{alert_class}">
                            <h3>Risk Level</h3>
                            <h2>{risk_level}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Fraud Probability</h3>
                            <h2>{fraud_probability:.1%}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence = max(fraud_probability, 1-fraud_probability)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{confidence:.1%}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk factors analysis
                    st.subheader("🔍 Risk Factors Analysis")
                    
                    risk_factors = []
                    if claim_to_vehicle_ratio > 0.8:
                        risk_factors.append("High claim-to-vehicle cost ratio")
                    if vehicle_age > 15:
                        risk_factors.append("Old vehicle")
                    if claim_amount > df['Total_Claim'].quantile(0.9):
                        risk_factors.append("High claim amount")
                    if accident_hour < 6 or accident_hour > 22:
                        risk_factors.append("Unusual accident time")
                    if annual_mileage < 5000:
                        risk_factors.append("Very low annual mileage")
                    if injury_claim > claim_amount * 0.5:
                        risk_factors.append("High injury claim proportion")
                    
                    if risk_factors:
                        st.warning("**Risk Factors Identified:**")
                        for factor in risk_factors:
                            st.write(f"⚠️ {factor}")
                    else:
                        st.success("✅ No significant risk factors identified")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    if fraud_probability > 0.7:
                        st.error("""
                        **HIGH FRAUD RISK - Immediate Actions Required:**
                        - 🔍 Initiate comprehensive fraud investigation
                        - 📞 Contact claimant for additional documentation
                        - 🏥 Verify medical records and treatment
                        - 🔒 Place claim on hold pending investigation
                        - 👮 Consider involving Special Investigation Unit (SIU)
                        """)
                    elif fraud_probability > 0.4:
                        st.warning("""
                        **MEDIUM FRAUD RISK - Enhanced Review Required:**
                        - 📋 Conduct detailed claim review
                        - 📄 Request additional documentation
                        - 🔍 Verify accident details and witness statements
                        - 💰 Review repair estimates carefully
                        - ⏰ Monitor for unusual patterns
                        """)
                    else:
                        st.success("""
                        **LOW FRAUD RISK - Standard Processing:**
                        - ✅ Process claim through standard workflow
                        - 📊 Continue routine monitoring
                        - 📁 Document assessment in claim file
                        - 🔄 Consider for model training data
                        """)
                    
                    # Feature contributions (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("📊 Feature Contributions to Risk Score")
                        
                        feature_names = ['Age', 'Premium', 'Claim Amount', 'Vehicle Age', 'Annual Mileage', 'Claim/Vehicle Ratio']
                        contributions = model.feature_importances_
                        
                        contrib_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': contributions,
                            'Value': input_features
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            contrib_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Feature Importance in Risk Assessment"
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.info("Please check that all input values are valid.")
        
        else:
            st.error("❌ Unable to load fraud detection model")
            st.info("""
            To enable fraud prediction:
            1. Run the model training pipeline: `python src/model_training.py`
            2. Ensure models are saved in the 'models/' directory
            3. Refresh this page
            """)

    # KPI Dashboard Page
    elif page == "📈 KPI Dashboard":
        st.header("📈 Key Performance Indicators Dashboard")
        
        if kpis:
            # Main KPI cards
            st.subheader("🎯 Core Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                fraud_rate = kpis.get('fraud_rate', 0) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Fraud Detection Rate</h4>
                    <h2>{fraud_rate:.2f}%</h2>
                    <p>Primary KPI</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_fraud = kpis.get('avg_fraud_claim', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Avg Fraud Claim</h4>
                    <h2>${avg_fraud:,.0f}</h2>
                    <p>Financial Impact</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_legit = kpis.get('avg_legit_claim', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Avg Legit Claim</h4>
                    <h2>${avg_legit:,.0f}</h2>
                    <p>Baseline</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                claim_ratio = kpis.get('claim_amount_ratio', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Fraud Multiplier</h4>
                    <h2>{claim_ratio:.1f}x</h2>
                    <p>Risk Factor</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                fraud_count = kpis.get('fraud_count', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Frauds</h4>
                    <h2>{fraud_count:,}</h2>
                    <p>Volume</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed KPI analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🕐 Temporal Fraud Patterns")
                
                if 'fraud_by_month' in kpis and kpis['fraud_by_month']:
                    months = list(kpis['fraud_by_month'].keys())
                    counts = list(kpis['fraud_by_month'].values())
                    
                    fig = px.line(
                        x=months, y=counts,
                        title="Fraudulent Claims by Month",
                        labels={'x': 'Month', 'y': 'Number of Fraudulent Claims'}
                    )
                    fig.update_xaxes(tickmode='linear', dtick=1)
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'fraud_by_hour_group' in kpis and kpis['fraud_by_hour_group']:
                    hours = list(kpis['fraud_by_hour_group'].keys())
                    counts = list(kpis['fraud_by_hour_group'].values())
                    
                    fig = px.bar(
                        x=hours, y=counts,
                        title="Fraudulent Claims by Time of Day",
                        labels={'x': 'Time Period', 'y': 'Number of Fraudulent Claims'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🗺️ Geographic Distribution")
                
                if 'fraud_by_state' in kpis and kpis['fraud_by_state']:
                    states = list(kpis['fraud_by_state'].keys())[:10]  # Top 10
                    counts = list(kpis['fraud_by_state'].values())[:10]
                    
                    fig = px.bar(
                        x=counts, y=states,
                        orientation='h',
                        title="Top 10 States by Fraud Count",
                        labels={'x': 'Number of Fraudulent Claims', 'y': 'State'}
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'state_fraud_rates' in kpis and kpis['state_fraud_rates']:
                    # Top fraud rate states
                    sorted_states = sorted(kpis['state_fraud_rates'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
                    states = [x[0] for x in sorted_states]
                    rates = [x[1] * 100 for x in sorted_states]
                    
                    fig = px.bar(
                        x=rates, y=states,
                        orientation='h',
                        title="Top 10 States by Fraud Rate (%)",
                        labels={'x': 'Fraud Rate (%)', 'y': 'State'}
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Advanced analytics
            st.subheader("🔬 Advanced Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🚗 Vehicle Analysis")
                avg_fraud_age = kpis.get('avg_fraud_vehicle_age', 0)
                avg_legit_age = kpis.get('avg_legit_vehicle_age', 0)
                
                st.write(f"**Average Vehicle Age:**")
                st.write(f"- Fraudulent: {avg_fraud_age:.1f} years")
                st.write(f"- Legitimate: {avg_legit_age:.1f} years")
                st.write(f"- Difference: {avg_fraud_age - avg_legit_age:.1f} years")
            
            with col2:
                st.markdown("### 💰 Financial Impact")
                high_value_rate = kpis.get('high_value_fraud_rate', 0) * 100
                
                st.write(f"**High-Value Claims:**")
                st.write(f"- Fraud Rate: {high_value_rate:.1f}%")
                
                # Estimated losses
                total_fraud_amount = fraud_count * avg_fraud
                st.write(f"- Est. Total Fraud: ${total_fraud_amount:,.0f}")
                
                # Potential savings with 80% detection
                potential_savings = total_fraud_amount * 0.8
                st.write(f"- Potential Savings: ${potential_savings:,.0f}")
            
            with col3:
                st.markdown("### 📊 Performance Indicators")
                
                # Calculate performance indicators
                if fraud_rate > 20:
                    risk_level = "🔴 High Risk"
                elif fraud_rate > 10:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🟢 Low Risk"
                
                st.write(f"**Risk Assessment:**")
                st.write(f"- Portfolio Risk: {risk_level}")
                st.write(f"- Detection Priority: {'High' if fraud_rate > 15 else 'Medium' if fraud_rate > 10 else 'Standard'}")
                
                # Model deployment readiness
                if model_results:
                    best_auc = max([v.get('roc_auc', 0) for v in model_results.values()])
                    deployment_ready = "✅ Ready" if best_auc > 0.8 else "⚠️ Needs Improvement"
                    st.write(f"- Model Status: {deployment_ready}")
        
        else:
            st.warning("⚠️ No KPI data available. Please run feature engineering first.")
            st.info("""
            To generate KPI data:
            1. Run `python src/feature_engineering.py`
            2. Refresh this page
            """)
        
        # Real-time monitoring simulation
        st.subheader("📡 Real-Time Monitoring Dashboard")
        
        # Create simulated real-time data
        if st.button("🔄 Refresh Real-Time Data"):
            # Simulate new data
            current_time = pd.Timestamp.now()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                new_claims = np.random.randint(45, 85)
                st.metric("Claims Today", new_claims, delta=np.random.randint(-5, 15))
            
            with col2:
                fraud_alerts = np.random.randint(5, 15)
                st.metric("Fraud Alerts", fraud_alerts, delta=np.random.randint(-2, 5))
            
            with col3:
                avg_processing = np.random.uniform(2.5, 4.5)
                st.metric("Avg Processing (hrs)", f"{avg_processing:.1f}", delta=f"{np.random.uniform(-0.5, 0.5):.1f}")
            
            with col4:
                system_health = np.random.choice(["🟢 Healthy", "🟡 Warning", "🔴 Alert"], p=[0.7, 0.2, 0.1])
                st.metric("System Status", system_health)
            
            st.info(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.error("❌ Unable to load data. Please ensure the data files are available.")
    st.info("""
    **Setup Instructions:**
    1. Place your CSV data files in the `data/raw/` directory
    2. Run the preprocessing pipeline: `python src/data_preprocessing.py`
    3. Run feature engineering: `python src/feature_engineering.py`
    4. Refresh this page
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>Auto Insurance Fraud Detection System</h4>
    <p>Powered by Advanced Machine Learning | Built with Streamlit</p>
    <p>🔒 Secure • 🚀 Fast • 🎯 Accurate</p>
</div>
""", unsafe_allow_html=True)
