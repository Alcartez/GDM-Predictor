import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="GDM Prediction Tool 2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
        color: #000000; /* Ensure text is dark on light background */
    }
    .high-risk {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

from sklearn.pipeline import Pipeline

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        
        # Load model files using paths relative to the script
        model_container = joblib.load(os.path.join(models_dir, 'best_lazy_model_LGBMClassifier.joblib'))
        
        # Extract classifier if it's a pipeline
        if isinstance(model_container, Pipeline):
            # Check for 'classifier' step name first (LazyPredict standard)
            if 'classifier' in model_container.named_steps:
                model = model_container.named_steps['classifier']
            else:
                # Fallback to last step
                model = model_container.steps[-1][1]
        else:
            model = model_container

        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        knn_imputer = joblib.load(os.path.join(models_dir, 'knn_imputer.joblib'))
        
        return model, scaler, feature_names, knn_imputer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Debug info for troubleshooting
        st.write(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
        return None, None, None, None

def predict_gdm(input_data, model, scaler, feature_names, imputer):
    """
    Make GDM prediction for a single patient
    """
    try:
        # Ensure input is in the correct format
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input must be a dictionary or DataFrame")
        
        # Ensure all feature columns are present
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Reorder columns
        input_df = input_df[feature_names]
        
        # Impute missing if needed (though UI prevents it usually, helpful for batch)
        # Note: In production, re-running fit_transform on single sample is wrong, 
        # but here we use the fitted imputer if available, or just handle NaNs.
        # The stored imputer is fitted.
        if input_df.isnull().any().any():
             input_array = imputer.transform(input_df)
             input_df = pd.DataFrame(input_array, columns=feature_names)

        # Scale features
        X_scaled = scaler.transform(input_df)
        
        # Convert back to DataFrame to preserve feature names
        # (LGBM/LazyPredict pipeline may expect a DataFrame)
        X_final = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Helper for binary classification
        # Some sklearn models expect dataframes, others numpy arrays. 
        # LGBM usually handles both, but scaler returns numpy.
        
        # Get Probabilities
        probabilities = model.predict_proba(X_final)[0]
        gdm_prob = probabilities[1]
        no_gdm_prob = probabilities[0]
        
        # Decision
        prediction = "GDM" if gdm_prob > 0.5 else "No GDM"
        risk_level = "High Risk" if gdm_prob > 0.5 else "Low Risk"
        
        return {
            'prediction': prediction,
            'gdm_probability': gdm_prob,
            'no_gdm_probability': no_gdm_prob,
            'risk_level': risk_level
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Load model
    model, scaler, feature_names, imputer = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check 'models/' folder.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🏥 GDM Prediction Tool 2.0</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Powered by LightGBM (Accuracy: 82.7% | AUC: 0.91)</p>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Options")
    prediction_mode = st.sidebar.selectbox("Select Mode:", ["Manual Input", "Batch Upload"])
    
    if prediction_mode == "Manual Input":
        st.header("👤 Manual Patient Input")
        
        with st.form("patient_input_form"):
            st.subheader("Patient Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Epigenetic Markers**")
                cg10139015 = st.number_input("cg10139015", 0.0, 1.0, 0.13, format="%.6f")
                cg23507676 = st.number_input("cg23507676", 0.0, 1.0, 0.60, format="%.6f")
                cg11001216 = st.number_input("cg11001216", 0.0, 1.0, 0.09, format="%.6f")
                cg04539775 = st.number_input("cg04539775", 0.0, 1.0, 0.69, format="%.6f")
                cg04985016 = st.number_input("cg04985016", 0.0, 1.0, 0.61, format="%.6f")
            
            with col2:
                st.markdown("**Clinical Information**")
                age = st.number_input("Age (years)", 15, 50, 28)
                hw_ratio = st.number_input("Height/Waist Ratio", 1.0, 3.0, 1.85, format="%.3f")
                fam_hist = st.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                hba1c = st.number_input("HbA1c Level", 4.0, 10.0, 5.2, format="%.1f")
                sys_bp = st.number_input("Systolic BP", 70, 180, 105)
                dia_bp = st.number_input("Diastolic BP", 40, 120, 70)
            
            submitted = st.form_submit_button("🔮 Predict Risk")
            
            if submitted:
                input_data = {
                    'cg10139015': cg10139015, 'cg23507676': cg23507676,
                    'cg11001216': cg11001216, 'cg04539775': cg04539775,
                    'cg04985016': cg04985016, 'Age_Yon': age,
                    'V1_Height_Waist_ratio': hw_ratio,
                    'V2_family_his_diab_Yon': fam_hist,
                    'V2_pt_of_hba1c_Yon': hba1c,
                    'V2_BP_rding1_sys_Yon': sys_bp,
                    'V2_BP_rding1_dia_Yon': dia_bp
                }
                
                with st.spinner("Analyzing..."):
                    result = predict_gdm(input_data, model, scaler, feature_names, imputer)
                
                if result:
                    st.header("Results")
                    # Risk Box
                    css_class = "high-risk" if result['gdm_probability'] > 0.5 else "low-risk"
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h3>{'🚨 HIGH RISK' if result['gdm_probability'] > 0.5 else '✅ LOW RISK'}</h3>
                        <p><strong>GDM Probability:</strong> {result['gdm_probability']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    fig = go.Figure(go.Bar(
                        x=[result['no_gdm_probability'], result['gdm_probability']],
                        y=['No GDM', 'GDM'],
                        orientation='h',
                        marker_color=['#4caf50', '#f44336']
                    ))
                    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True)

    elif prediction_mode == "Batch Upload":
        st.header("📂 Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} records")
            
            if st.button("Predict All"):
                predictions = []
                bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    res = predict_gdm(row.to_dict(), model, scaler, feature_names, imputer)
                    if res:
                        predictions.append({
                            'Patient_ID': idx + 1,
                            'Prediction': res['prediction'],
                            'GDM_Probability': res['gdm_probability'],
                            'Risk_Level': res['risk_level']
                        })
                    bar.progress((idx+1)/len(df))
                
                res_df = pd.DataFrame(predictions)
                st.dataframe(res_df)
                st.download_button("Download CSV", res_df.to_csv(index=False), "results.csv", "text/csv")

if __name__ == "__main__":
    main()
