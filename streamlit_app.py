import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="GDM Prediction Tool",
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

class GraphSAGE(nn.Module):
    """
    GraphSAGE Neural Network for GDM Prediction
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=2, dropout=0.2):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x, edge_index):
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        
        return F.log_softmax(x, dim=1)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        # Load model - Updated paths for GitHub deployment
        checkpoint = torch.load('models/graphsage_model.pth', map_location='cpu')
        
        # Initialize model
        config = checkpoint['model_config']
        model = GraphSAGE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            dropout=config['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load preprocessing objects - Updated paths for GitHub deployment
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        return model, scaler, feature_names, checkpoint['test_metrics']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def create_patient_graph(X, k_neighbors=5):
    """
    Create patient similarity graph using KNN
    """
    n_samples = len(X)
    
    # Handle single sample case
    if n_samples == 1:
        # For single patient, create a self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return edge_index
    
    # For multiple samples, use KNN
    max_k = min(k_neighbors, n_samples - 1)
    knn = NearestNeighbors(n_neighbors=max_k + 1, metric='euclidean')
    knn.fit(X)
    
    distances, indices = knn.kneighbors(X)
    
    # Create edge list (excluding self-connections)
    edge_index = []
    for i in range(n_samples):
        for j in range(1, max_k + 1):
            if j < len(indices[i]):
                edge_index.append([i, indices[i][j]])
                edge_index.append([indices[i][j], i])  # Make it undirected
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def predict_gdm(input_data, model, scaler, feature_names):
    """
    Make GDM prediction for a single patient
    """
    try:
        # Ensure input is in the correct format
        if isinstance(input_data, dict):
            # Convert dict to DataFrame
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input must be a dictionary or DataFrame")
        
        # Ensure all feature columns are present and in the right order
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale the features
        X_scaled = scaler.transform(input_df)
        
        # Create graph structure for the input
        edge_index = create_patient_graph(X_scaled, k_neighbors=5)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(X_tensor, edge_index)
            probabilities = torch.exp(output)
            predicted_class = torch.argmax(output, dim=1)
            gdm_probability = probabilities[:, 1].item()
        
        return {
            'prediction': 'GDM' if predicted_class.item() == 1 else 'No GDM',
            'gdm_probability': gdm_probability,
            'no_gdm_probability': probabilities[:, 0].item(),
            'risk_level': 'High Risk' if gdm_probability > 0.5 else 'Low Risk'
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Load model
    model, scaler, feature_names, test_metrics = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 GDM Prediction Tool</h1>', unsafe_allow_html=True)
    
    # Model performance display
    if test_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{test_metrics['accuracy']:.1%}")
        with col2:
            st.metric("ROC AUC", f"{test_metrics['auc']:.3f}")
        with col3:
            st.metric("Features Used", len(feature_names))
    
    # Sidebar for input mode selection
    st.sidebar.title("📋 Options")
    prediction_mode = st.sidebar.selectbox(
        "Select Prediction Mode:",
        ["Manual Input", "Batch Upload"]
    )
    
    if prediction_mode == "Manual Input":
        st.header("👤 Manual Patient Input")
        
        # Create input form
        with st.form("patient_input_form"):
            st.subheader("Patient Information")
            
            # Group inputs by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Epigenetic Markers**")
                cg10139015 = st.number_input("cg10139015", min_value=0.0, max_value=1.0, value=0.13, format="%.6f")
                cg23507676 = st.number_input("cg23507676", min_value=0.0, max_value=1.0, value=0.60, format="%.6f")
                cg11001216 = st.number_input("cg11001216", min_value=0.0, max_value=1.0, value=0.09, format="%.6f")
                cg04539775 = st.number_input("cg04539775", min_value=0.0, max_value=1.0, value=0.69, format="%.6f")
                cg04985016 = st.number_input("cg04985016", min_value=0.0, max_value=1.0, value=0.61, format="%.6f")
            
            with col2:
                st.markdown("**Clinical Information**")
                age = st.number_input("Age (years)", min_value=15, max_value=50, value=28)
                height_waist_ratio = st.number_input("Height/Waist Ratio", min_value=1.0, max_value=3.0, value=1.85, format="%.3f")
                family_diabetes = st.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                hba1c = st.number_input("HbA1c Level", min_value=4.0, max_value=10.0, value=5.2, format="%.1f")
                bp_systolic = st.number_input("Systolic BP (mmHg)", value=105, min_value=70, max_value=180)
                bp_diastolic = st.number_input("Diastolic BP (mmHg)", value=70, min_value=40, max_value=120)
            
            submitted = st.form_submit_button("🔮 Predict GDM Risk", use_container_width=True)
            
            if submitted:
                # Create input dictionary
                input_data = {
                    'cg10139015': cg10139015,
                    'cg23507676': cg23507676,
                    'cg11001216': cg11001216,
                    'cg04539775': cg04539775,
                    'cg04985016': cg04985016,
                    'Age_Yon': age,
                    'V1_Height_Waist_ratio': height_waist_ratio,
                    'V2_family_his_diab_Yon': family_diabetes,
                    'V2_pt_of_hba1c_Yon': hba1c,
                    'V2_BP_rding1_sys_Yon': bp_systolic,
                    'V2_BP_rding1_dia_Yon': bp_diastolic
                }
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    result = predict_gdm(input_data, model, scaler, feature_names)
                
                if result:
                    st.header("📊 Prediction Results")
                    
                    # Display results in a nice format
                    if result['risk_level'] == 'High Risk':
                        st.markdown(f"""
                        <div class="prediction-box high-risk">
                            <h3>🚨 HIGH RISK - GDM Prediction</h3>
                            <p><strong>Prediction:</strong> {result['prediction']}</p>
                            <p><strong>GDM Probability:</strong> {result['gdm_probability']:.1%}</p>
                            <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box low-risk">
                            <h3>✅ LOW RISK - No GDM Prediction</h3>
                            <p><strong>Prediction:</strong> {result['prediction']}</p>
                            <p><strong>GDM Probability:</strong> {result['gdm_probability']:.1%}</p>
                            <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create probability chart
                    fig = go.Figure(go.Bar(
                        x=[result['no_gdm_probability'], result['gdm_probability']],
                        y=['No GDM', 'GDM'],
                        orientation='h',
                        marker_color=['#4caf50', '#f44336']
                    ))
                    fig.update_layout(
                        title="Prediction Probabilities",
                        xaxis_title="Probability",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Clinical recommendations
                    st.header("📋 Clinical Recommendations")
                    if result['gdm_probability'] > 0.5:
                        st.warning("""
                        **High Risk Recommendations:**
                        - Consider early glucose screening
                        - Monitor blood glucose levels closely
                        - Recommend dietary counseling
                        - Schedule follow-up appointments
                        - Consider prophylactic interventions
                        """)
                    else:
                        st.info("""
                        **Low Risk Recommendations:**
                        - Continue routine prenatal care
                        - Standard glucose screening at 24-28 weeks
                        - Maintain healthy lifestyle
                        - Regular monitoring as per standard protocol
                        """)

    elif prediction_mode == "Batch Upload":
        st.header("📂 Batch Prediction")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with patient data",
            type=['csv'],
            help="CSV should contain the same features used in training"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} patient records")
                
                # Display data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Check if required columns are present
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    return
                
                # Make predictions for all patients
                if st.button("🔮 Predict All Patients", use_container_width=True):
                    predictions = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        # Convert row to dict
                        input_data = row.to_dict()
                        
                        # Make prediction
                        result = predict_gdm(input_data, model, scaler, feature_names)
                        
                        if result:
                            predictions.append({
                                'Patient_ID': idx + 1,
                                'Prediction': result['prediction'],
                                'GDM_Probability': result['gdm_probability'],
                                'Risk_Level': result['risk_level']
                            })
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing patient {idx + 1} of {len(df)}")
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(predictions)
                    
                    # Display results
                    st.header("📊 Batch Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Patients", len(results_df))
                        high_risk_count = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                        st.metric("High Risk Patients", high_risk_count)
                    
                    with col2:
                        avg_gdm_prob = results_df['GDM_Probability'].mean()
                        st.metric("Average GDM Probability", f"{avg_gdm_prob:.1%}")
                        
                    # Display detailed results
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="gdm_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("📊 Risk Distribution")
                    
                    # Risk level distribution
                    risk_counts = results_df['Risk_Level'].value_counts()
                    fig1 = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Probability distribution
                    fig2 = px.histogram(
                        results_df,
                        x='GDM_Probability',
                        nbins=20,
                        title="GDM Probability Distribution"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🏥 GDM Prediction Tool | Developed using GraphSAGE Graph Neural Networks</p>
        <p><small>For clinical use only. Not a substitute for professional medical advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
