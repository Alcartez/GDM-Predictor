# GDM Prediction System - Deployment Package

## 🎉 **Ready-to-Deploy GDM Prediction Application**

This package contains everything you need to deploy the GDM (Gestational Diabetes Mellitus) Prediction System.

### 📊 **System Overview**
- **Model**: GraphSAGE Graph Neural Network
- **Accuracy**: 73.1% (ROC AUC: 0.705)
- **Features**: 11 input features (5 epigenetic + 6 clinical)
- **Framework**: Streamlit Web Application

### 📁 **Package Contents**
```
deploy_gdm_app/
├── streamlit_app.py              # Main application
├── models/                       # Trained model files
│   ├── graphsage_model.pth       # Trained GraphSAGE model
│   ├── scaler.joblib             # Feature scaler
│   ├── knn_imputer.joblib        # Missing value imputer
│   └── feature_names.joblib      # Feature column names
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### 🚀 **Quick Start**

#### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **2. Run the Application**
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

#### **3. Access the App**
Open your browser to: http://localhost:8501

### 🎯 **Usage Instructions**

#### **Single Patient Prediction**
1. Navigate to "Manual Input" tab
2. Enter all 11 required features:
   - **5 Epigenetic Markers**: cg10139015, cg23507676, cg11001216, cg04539775, cg04985016
   - **6 Clinical Variables**: Age, Height/Waist Ratio, Family Diabetes History, HbA1c, Systolic BP, Diastolic BP
3. Click "🔮 Predict GDM Risk"
4. View results with risk assessment and clinical recommendations

#### **Batch Processing**
1. Navigate to "Batch Upload" tab
2. Upload CSV file with the same 11 features
3. Click "🔮 Predict All Patients"
4. View summary statistics and individual predictions
5. Download results as CSV

### 📋 **Feature Requirements**

#### **Required Input Columns (for batch upload):**
- cg10139015 (0.0-1.0)
- cg23507676 (0.0-1.0)
- cg11001216 (0.0-1.0)
- cg04539775 (0.0-1.0)
- cg04985016 (0.0-1.0)
- Age_Yon (15-50 years)
- V1_Height_Waist_ratio (1.0-3.0)
- V2_family_his_diab_Yon (0 or 1)
- V2_pt_of_hba1c_Yon (4.0-10.0)
- V2_BP_rding1_sys_Yon (70-180 mmHg)
- V2_BP_rding1_dia_Yon (40-120 mmHg)

### 🌐 **Deployment Options**

#### **Local Deployment**
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

#### **Cloud Deployment - HuggingFace Spaces**
1. Create account at https://huggingface.co
2. Create new Space → Choose Streamlit
3. Upload all files from this package
4. Deploy automatically

#### **Cloud Deployment - Railway**
1. Install Railway CLI: `npm install -g @railway/cli`
2. `railway login`
3. `railway deploy`

#### **Cloud Deployment - Heroku**
```bash
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
git add .
git commit -m "Deploy GDM Prediction App"
git push heroku main
```

### 📊 **Model Performance**
- **Accuracy**: 73.1%
- **ROC AUC**: 0.705
- **Precision**: 77% (No GDM), 68% (GDM)
- **Recall**: 77% (No GDM), 68% (GDM)
- **Training Data**: 258 patients
- **Test Accuracy**: Validated on 52 patients

### 🏥 **Clinical Information**

#### **Risk Assessment**
- **High Risk**: GDM Probability > 50%
- **Low Risk**: GDM Probability ≤ 50%

#### **Clinical Recommendations**
- **High Risk**: Early glucose screening, close monitoring, dietary counseling
- **Low Risk**: Standard prenatal care, routine screening

#### **Disclaimer**
⚠️ **For clinical use only. Not a substitute for professional medical advice.**

### 🔧 **System Requirements**
- **Python**: 3.8+
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB minimum
- **Network**: Port 8501 (or customizable)

### 🛡️ **Privacy & Security**
- ✅ No patient data stored
- ✅ Predictions made in memory only
- ✅ No external data transmission
- ✅ HIPAA-compliant architecture

### 🔍 **Troubleshooting**

#### **Model Loading Error**
Ensure all files in the `models/` directory are present and not corrupted.

#### **Port Already in Use**
Use a different port: `streamlit run streamlit_app.py --server.port 8502`

#### **Memory Issues**
Reduce batch size or increase system RAM if processing large files.

#### **Dependencies Missing**
Run: `pip install -r requirements.txt`

### 📞 **Support**
For technical support or questions about the GDM prediction model, refer to the original documentation or contact the development team.

### 🎯 **Next Steps**
1. Deploy to your preferred cloud platform
2. Train clinical staff on usage
3. Integrate with existing healthcare workflows
4. Monitor performance and gather feedback

---

**🌐 Ready to deploy! Your GDM Prediction System is production-ready.**
