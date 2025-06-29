import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 1rem;
        color: #f0f0f0;
    }
    .prediction-high {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-medium {
        background: linear-gradient(90deg, #ffa726 0%, #fb8c00 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-low {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models_and_data():
    """Load trained models and associated data"""
    try:
        # Load models
        models = {}
        model_files = {
            'Random Forest': 'models/saved_models/random_forest.joblib',
            'Logistic Regression': 'models/saved_models/logistic_regression.joblib',
            'XGBoost': 'models/saved_models/xgboost.joblib'
        }
        
        for name, path in model_files.items():
            try:
                if os.path.exists(path):
                    models[name] = joblib.load(path)
                    st.sidebar.success(f"✅ {name} loaded")
            except Exception as e:
                st.sidebar.warning(f"⚠️ {name} not available")
                continue
        
        # Load scaler
        scaler = None
        try:
            scaler = joblib.load('models/saved_models/scaler.joblib')
            st.sidebar.success("✅ Scaler loaded")
        except:
            st.sidebar.warning("⚠️ Scaler not found")
        
        # Load feature names
        feature_names = []
        try:
            with open('models/saved_models/feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            st.sidebar.success(f"✅ {len(feature_names)} features loaded")
        except:
            st.sidebar.error("❌ Feature names not found")
        
        # Load model results
        results = {}
        try:
            with open('models/saved_models/model_results.json', 'r') as f:
                results = json.load(f)
            st.sidebar.success("✅ Model results loaded")
        except:
            st.sidebar.warning("⚠️ Model results not found")
            results = {'best_model': list(models.keys())[0] if models else 'Random Forest'}
        
        return models, scaler, feature_names, results
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def create_customer_input_form():
    """Create input form for customer data"""
    st.sidebar.header("🔮 Customer Information")
    
    # Demographics
    st.sidebar.subheader("👥 Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    
    # Account Info
    st.sidebar.subheader("📅 Account Information")
    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 
                                          min_value=monthly_charges, 
                                          value=monthly_charges * tenure,
                                          step=10.0)
    
    # Services
    st.sidebar.subheader("📱 Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    
    if phone_service == "Yes":
        multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"
    
    internet_service = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    
    if internet_service != "No":
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = online_backup = device_protection = tech_support = "No internet service"
        streaming_tv = streaming_movies = "No internet service"
    
    # Contract & Payment
    st.sidebar.subheader("📄 Contract & Payment")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    # Create customer data dictionary
    customer_data = {
        'customerID': 'CUST_PRED_001',
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return customer_data

def engineer_features(customer_data):
    """Apply the same feature engineering as training - ROBUST VERSION"""
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Handle TotalCharges conversion (in case it's string)
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    
    # 1. Binary Categorical Features (Yes/No)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map({'No': 0, 'Yes': 1})
    
    # 2. Internet Service - CREATE ALL POSSIBLE COLUMNS
    internet_categories = ['DSL', 'Fiber optic', 'No']
    for category in internet_categories:
        df[f'Internet_{category}'] = (df['InternetService'] == category).astype(int)
    
    # Internet service risk score
    internet_risk = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    df['InternetService_risk'] = df['InternetService'].map(internet_risk)
    
    # 3. Contract - CREATE ALL POSSIBLE COLUMNS
    contract_categories = ['Month-to-month', 'One year', 'Two year']
    for category in contract_categories:
        df[f'Contract_{category}'] = (df['Contract'] == category).astype(int)
    
    # Contract stability score
    contract_stability = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    df['Contract_stability'] = df['Contract'].map(contract_stability)
    
    # 4. Payment Method - CREATE ALL POSSIBLE COLUMNS
    payment_categories = ['Bank transfer (automatic)', 'Credit card (automatic)', 
                         'Electronic check', 'Mailed check']
    for category in payment_categories:
        df[f'Payment_{category}'] = (df['PaymentMethod'] == category).astype(int)
    
    # Payment method risk score
    payment_risk = {
        'Credit card (automatic)': 1,
        'Bank transfer (automatic)': 1,
        'Mailed check': 2,
        'Electronic check': 3
    }
    df['PaymentMethod_risk'] = df['PaymentMethod'].map(payment_risk)
    
    # 5. Service columns with "No internet service" or "No phone service"
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    
    for col in service_cols:
        if col in df.columns:
            # Create binary: has service (1) vs doesn't have service (0)
            df[f'{col}_binary'] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
            
            # Create availability: service available (1) vs not available (0)
            df[f'{col}_available'] = df[col].apply(
                lambda x: 0 if 'No internet service' in str(x) or 'No phone service' in str(x) else 1
            )
    
    # 6. Tenure-based features
    def categorize_tenure(tenure):
        if tenure <= 12:
            return 'New'
        elif tenure <= 24:
            return 'Medium'
        elif tenure <= 48:
            return 'Long'
        else:
            return 'Loyal'
    
    df['tenure_category'] = df['tenure'].apply(categorize_tenure)
    
    # Create ALL tenure category columns
    tenure_categories = ['Long', 'Loyal', 'Medium', 'New']
    for category in tenure_categories:
        df[f'Tenure_{category}'] = (df['tenure_category'] == category).astype(int)
    
    # Additional tenure features
    df['tenure_years'] = df['tenure'] / 12
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
    
    # 7. Charges-based features
    # Average monthly charge (total/tenure, but handle tenure=0)
    df['avg_monthly_charge'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    # Charge categories
    def categorize_charges(charges):
        if charges <= 35:
            return 'Low'
        elif charges <= 65:
            return 'Medium'
        else:
            return 'High'
    
    df['charges_category'] = df['MonthlyCharges'].apply(categorize_charges)
    
    # Create ALL charges category columns
    charges_categories = ['High', 'Low', 'Medium']
    for category in charges_categories:
        df[f'Charges_{category}'] = (df['charges_category'] == category).astype(int)
    
    # Price sensitivity indicators
    df['high_monthly_charges'] = (df['MonthlyCharges'] > 80).astype(int)
    df['low_monthly_charges'] = (df['MonthlyCharges'] < 35).astype(int)
    
    # 8. Service count features
    service_binary_cols = [col for col in df.columns if col.endswith('_binary')]
    df['total_services'] = df[service_binary_cols].sum(axis=1)
    
    # Service adoption rate (services used / services available)
    service_available_cols = [col for col in df.columns if col.endswith('_available')]
    df['services_available'] = df[service_available_cols].sum(axis=1)
    df['service_adoption_rate'] = np.where(
        df['services_available'] > 0,
        df['total_services'] / df['services_available'],
        0
    )
    
    # Premium services (streaming)
    premium_services = ['StreamingTV_binary', 'StreamingMovies_binary']
    if all(col in df.columns for col in premium_services):
        df['premium_services'] = df[premium_services].sum(axis=1)
        df['has_premium'] = (df['premium_services'] > 0).astype(int)
    
    # Protection services
    protection_services = ['OnlineSecurity_binary', 'OnlineBackup_binary', 'DeviceProtection_binary', 'TechSupport_binary']
    if all(col in df.columns for col in protection_services):
        df['protection_services'] = df[protection_services].sum(axis=1)
        df['has_protection'] = (df['protection_services'] > 0).astype(int)
    
    # 9. Advanced feature engineering
    # Risk score combinations
    risk_factors = ['Contract_stability', 'PaymentMethod_risk', 'InternetService_risk']
    for factor in risk_factors:
        if factor in df.columns:
            max_val = {'Contract_stability': 3, 'PaymentMethod_risk': 3, 'InternetService_risk': 2}[factor]
            min_val = 1
            df[f'{factor}_normalized'] = (df[factor] - min_val) / (max_val - min_val)
    
    # Composite risk score
    normalized_factors = [f'{factor}_normalized' for factor in risk_factors if f'{factor}_normalized' in df.columns]
    if normalized_factors:
        df['composite_risk_score'] = df[normalized_factors].mean(axis=1)
    
    # 10. Customer lifecycle features
    # Customer lifetime value estimate (tenure * monthly charges)
    df['estimated_clv'] = df['tenure'] * df['MonthlyCharges']
    
    # Revenue per month of tenure
    df['revenue_per_tenure_month'] = df['TotalCharges'] / np.maximum(df['tenure'], 1)
    
    # Spending trajectory (current vs average)
    df['spending_above_avg'] = (df['MonthlyCharges'] > df['avg_monthly_charge']).astype(int)
    
    # 11. Demographic and behavioral combinations
    # Senior citizen + contract combination
    if 'SeniorCitizen' in df.columns and 'Contract_stability' in df.columns:
        df['senior_short_contract'] = ((df['SeniorCitizen'] == 1) & (df['Contract_stability'] == 1)).astype(int)
    
    # Family indicators
    family_cols = ['Partner_encoded', 'Dependents_encoded']
    if all(col in df.columns for col in family_cols):
        df['family_size'] = df[family_cols].sum(axis=1)
        df['has_family'] = (df['family_size'] > 0).astype(int)
    
    # Phone + Internet service combination (FIXED)
    phone_service_encoded = df.get('PhoneService_encoded', pd.Series([0])).iloc[0]
    internet_dsl = df.get('Internet_DSL', pd.Series([0])).iloc[0]
    internet_fiber = df.get('Internet_Fiber optic', pd.Series([0])).iloc[0]
    
    df['full_service_customer'] = ((phone_service_encoded == 1) & 
                                  ((internet_dsl == 1) | (internet_fiber == 1))).astype(int)
    
    return df

def predict_churn(customer_data, models, scaler, feature_names):
    """Make churn prediction for customer"""
    if not models or not feature_names:
        st.error("Models or feature names not loaded properly")
        return None, None
    
    try:
        # Engineer features
        df_engineered = engineer_features(customer_data)
        
        # Create a DataFrame with all required features, filling missing ones with 0
        X_customer = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the features that exist in the engineered data
        for feature in feature_names:
            if feature in df_engineered.columns:
                X_customer[feature] = df_engineered[feature].iloc[0]
            # Missing features will remain 0 (default value)
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            try:
                if model_name == 'Logistic Regression' and scaler is not None:
                    # Scale features for logistic regression
                    X_scaled = scaler.transform(X_customer)
                    prob = model.predict_proba(X_scaled)[0]
                    pred = model.predict(X_scaled)[0]
                else:
                    # Tree-based models don't need scaling
                    prob = model.predict_proba(X_customer)[0]
                    pred = model.predict(X_customer)[0]
                
                predictions[model_name] = pred
                probabilities[model_name] = prob[1]  # Probability of churn
                
            except Exception as e:
                st.warning(f"Error with {model_name}: {str(e)}")
                continue
        
        return predictions, probabilities
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def create_risk_assessment(customer_data, churn_probability):
    """Create risk assessment and recommendations"""
    risk_factors = []
    
    # High-risk factors
    if customer_data['Contract'] == 'Month-to-month':
        risk_factors.append("❌ Month-to-month contract (high churn risk)")
    
    if customer_data['PaymentMethod'] == 'Electronic check':
        risk_factors.append("❌ Electronic check payment (high churn risk)")
    
    if customer_data['tenure'] <= 12:
        risk_factors.append("❌ Low tenure - new customer (high churn risk)")
    
    if customer_data['MonthlyCharges'] > 80:
        risk_factors.append("❌ High monthly charges (price sensitivity risk)")
    
    if customer_data['TechSupport'] == 'No':
        risk_factors.append("❌ No tech support (satisfaction risk)")
    
    if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
        risk_factors.append("❌ No online security (service gap)")
    
    # Protective factors
    protective_factors = []
    
    if customer_data['Contract'] in ['One year', 'Two year']:
        protective_factors.append("✅ Long-term contract")
    
    if customer_data['Dependents'] == 'Yes':
        protective_factors.append("✅ Has dependents (stability factor)")
    
    if customer_data['tenure'] > 24:
        protective_factors.append("✅ Loyal customer (high tenure)")
    
    if customer_data['TechSupport'] == 'Yes':
        protective_factors.append("✅ Has tech support")
    
    if customer_data['PaymentMethod'] in ['Credit card (automatic)', 'Bank transfer (automatic)']:
        protective_factors.append("✅ Automatic payment method")
    
    # Recommendations
    recommendations = []
    
    if churn_probability > 0.7:
        recommendations.extend([
            "🚨 HIGH PRIORITY: Immediate intervention required",
            "📞 Personal outreach from customer success team",
            "💰 Consider special retention offer or discount",
            "🔧 Provide additional support and service optimization",
            "📋 Review account for service gaps and upgrade opportunities"
        ])
    elif churn_probability > 0.5:
        recommendations.extend([
            "⚠️ MEDIUM PRIORITY: Proactive engagement recommended",
            "📧 Send targeted retention campaign",
            "🎁 Offer service upgrades or loyalty rewards",
            "📊 Monitor usage patterns closely",
            "💬 Reach out for satisfaction survey"
        ])
    elif churn_probability > 0.3:
        recommendations.extend([
            "👀 WATCH LIST: Monitor for changes",
            "🌟 Consider upselling opportunities",
            "📈 Track engagement metrics",
            "💎 Maintain service quality"
        ])
    else:
        recommendations.extend([
            "✅ LOW RISK: Standard monitoring sufficient",
            "🌟 Excellent candidate for upselling",
            "🏆 Use as reference for customer satisfaction",
            "💎 Maintain excellent service quality"
        ])
    
    return risk_factors, protective_factors, recommendations

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict customer churn probability and get actionable business insights")
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        models, scaler, feature_names, results = load_models_and_data()
    
    if not models:
        st.error("❌ Could not load models. Please ensure model files are in the correct location.")
        st.info("Expected model files in: models/saved_models/")
        return
    
    # Sidebar for input
    customer_data = create_customer_input_form()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔮 Prediction Results")
        
        if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                # Make predictions
                predictions, probabilities = predict_churn(customer_data, models, scaler, feature_names)
                
                if predictions and probabilities:
                    # Get best model prediction
                    best_model = results.get('best_model', list(models.keys())[0])
                    best_prob = probabilities.get(best_model, list(probabilities.values())[0])
                    
                    # Display main prediction
                    if best_prob > 0.7:
                        st.markdown(f'<div class="prediction-high">🚨 HIGH CHURN RISK<br>Probability: {best_prob:.1%}</div>', 
                                  unsafe_allow_html=True)
                    elif best_prob > 0.5:
                        st.markdown(f'<div class="prediction-medium">⚠️ MEDIUM CHURN RISK<br>Probability: {best_prob:.1%}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-low">✅ LOW CHURN RISK<br>Probability: {best_prob:.1%}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Model comparison
                    st.subheader("📊 Model Predictions Comparison")
                    prob_df = pd.DataFrame({
                        'Model': list(probabilities.keys()),
                        'Churn Probability': [f"{prob:.1%}" for prob in probabilities.values()],
                        'Probability Value': list(probabilities.values())
                    })
                    
                    st.dataframe(prob_df[['Model', 'Churn Probability']], use_container_width=True)
                    
                    # Probability visualization
                    fig = px.bar(prob_df, x='Model', y='Probability Value', 
                               title='Churn Probability by Model',
                               color='Probability Value',
                               color_continuous_scale=['green', 'yellow', 'red'])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store prediction for risk assessment
                    st.session_state['prediction_made'] = True
                    st.session_state['churn_probability'] = best_prob
                    st.session_state['customer_data'] = customer_data
                else:
                    st.error("Failed to make prediction. Please check your input and try again.")
    
    with col2:
        st.header("📈 Risk Assessment & Recommendations")
        
        if st.session_state.get('prediction_made', False):
            churn_prob = st.session_state['churn_probability']
            customer_data = st.session_state['customer_data']
            
            risk_factors, protective_factors, recommendations = create_risk_assessment(customer_data, churn_prob)
            
            # Risk factors
            if risk_factors:
                st.subheader("⚠️ Risk Factors")
                for factor in risk_factors:
                    st.write(factor)
            
            # Protective factors
            if protective_factors:
                st.subheader("🛡️ Protective Factors")
                for factor in protective_factors:
                    st.write(factor)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.write(rec)
            
            # Customer profile summary
            st.subheader("👤 Customer Profile")
            profile_data = {
                'Tenure': f"{customer_data['tenure']} months",
                'Monthly Charges': f"${customer_data['MonthlyCharges']:.2f}",
                'Contract': customer_data['Contract'],
                'Payment Method': customer_data['PaymentMethod'],
                'Internet Service': customer_data['InternetService']
            }
            
            for key, value in profile_data.items():
                st.write(f"**{key}:** {value}")
        
        else:
            st.info("👈 Enter customer information and click 'Predict Churn' to see risk assessment and recommendations.")
    
    # Model performance section
    st.header("📊 Model Performance Overview")
    
    if results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Best Model",
                value=results.get('best_model', 'N/A'),
                delta="Selected"
            )
        
        with col2:
            test_auc = results.get('business_metrics', {}).get('test_auc', 0)
            st.metric(
                label="Test AUC",
                value=f"{test_auc:.3f}",
                delta="Excellent" if test_auc > 0.8 else "Good"
            )
        
        with col3:
            test_acc = results.get('business_metrics', {}).get('test_accuracy', 0)
            st.metric(
                label="Accuracy",
                value=f"{test_acc:.1%}",
                delta="High Performance"
            )
        
        with col4:
            precision = results.get('business_metrics', {}).get('precision', 0)
            st.metric(
                label="Precision",
                value=f"{precision:.1%}",
                delta="Reliable Predictions"
            )
        
        # Feature importance
        if 'feature_importance' in results:
            st.header("🎯 Top Feature Importance")
            
            importance_df = pd.DataFrame(results['feature_importance'][:10])
            
            fig = px.bar(importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features for Churn Prediction')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 Built with Machine Learning")
    st.markdown("This application uses advanced ML algorithms to predict customer churn and provide actionable business insights.")
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.write("**Models Used:**")
        for model_name in models.keys():
            st.write(f"- {model_name}")
        
        st.write(f"**Features:** {len(feature_names)} engineered features")
        st.write("**Training Data:** Customer dataset with churn labels")
        st.write("**Evaluation:** Cross-validation with stratified sampling")
        
        if st.checkbox("Show Feature Names"):
            st.write("**All Features:**")
            for i, feature in enumerate(feature_names, 1):
                st.write(f"{i}. {feature}")

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False

if __name__ == "__main__":
    main()