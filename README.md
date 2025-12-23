# GDM Prediction System - Deployment Package 2.0 (Optimized)

## 🚀 **High-Performance GDM Prediction App**

This is the **optimized version** of the Gestational Diabetes Mellitus (GDM) Prediction System. It uses a **LightGBM** model instead of GraphSAGE, resulting in significantly higher accuracy and a much lighter deployment footprint.

### 📊 **Key Improvements**
| Metric | Old Version (v1) | **New Version (v2)** |
| :--- | :--- | :--- |
| **Model** | GraphSAGE (GNN) | **LightGBM (Gradient Boosting)** |
| **Accuracy** | 73.1% | **82.7%** |
| **ROC AUC** | 0.705 | **0.914** |
| **Install Size** | ~2.5 GB (PyTorch) | **~250 MB** (Lightweight) |
| **Speed** | Moderate | **Blazing Fast** |

---

### 📁 **Package Contents**
```
deploy_gdm_app_2/
├── streamlit_app.py              # Main application
├── requirements.txt              # Minimal dependencies
├── models/                       # Optimized artifacts
│   ├── best_lazy_model_LGBMClassifier.joblib   # The Brain (82.7% Acc)
│   ├── scaler.joblib             # Data Scaler
│   ├── knn_imputer.joblib        # Missing Data Handler
│   └── feature_names.joblib      # Column Definitions
└── README.md                     # This file
```

---

### 🚀 **Quick Start**

#### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **2. Run the Application**
```bash
streamlit run streamlit_app.py
```
*App will open at: http://localhost:8501*

---

### 🏥 **Usage Guide**

#### **Manual Input Mode**
1. Select "Manual Input" from the sidebar.
2. Enter the patient's **5 Epigenetic Markers** and **6 Clinical Variables**.
    *   *Tip: Hover over fields for allowed ranges (e.g., Age 15-50).*
3. Click **"🔮 Predict Risk"**.
4. Receive an instant specific risk assessment (High/Low) and probability score.

#### **Batch Upload Mode**
1. Select "Batch Upload" from the sidebar.
2. Upload a CSV file containing patient data.
3. Click "Predict All".
4. Analyze the summary dashboard and **download results** as a CSV.

---

### 🛡️ **Technical Notes**
- **Privacy Check**: This app runs entirely locally (or on your private server). No data is sent to external APIs.
- **Missing Data**: The system automatically imputes missing values using a KNN imputer (trained on the original dataset) to ensure robust predictions even with incomplete records.

---

**Developed for Clinical Research Support.**
*Improvement Iteration: December 2025*
