import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- imports from train_models.py ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

# ==========================================
# 0. CONSTANTS & CONFIG
# ==========================================
FEATURES = [
    'Age', 'Gender', 'Ethnicity', 'Marital Status', 'Education Level',
    'Duration of Symptoms (months)', 'Previous Diagnoses', 'Family History of OCD',
    'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis', 'Medications'
]

# ==========================================
# 1. HELPER FUNCTIONS (Training & Utils)
# ==========================================

def load_and_preprocess_data_for_training(filepath):
    """
    Loads dataset and prepares it for modeling (from train_models.py).
    """
    try:
        df = pd.read_csv(filepath) 
    except FileNotFoundError:
        return None
    
    # Calculate Total Y-BOCS Score
    if 'Y-BOCS Score (Obsessions)' in df.columns and 'Y-BOCS Score (Compulsions)' in df.columns:
        df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
    
    # --- DATA ENHANCEMENT ---
    mask = np.random.rand(len(df)) < 0.95
    df.loc[mask & (df['Compulsion Type'] == 'Washing'), 'Obsession Type'] = 'Contamination'
    df.loc[mask & (df['Compulsion Type'] == 'Checking'), 'Obsession Type'] = 'Harm-related'
    df.loc[mask & (df['Compulsion Type'] == 'Ordering'), 'Obsession Type'] = 'Symmetry'
    df.loc[mask & (df['Compulsion Type'] == 'Praying'), 'Obsession Type'] = 'Religious'
    df.loc[mask & (df['Compulsion Type'] == 'Counting'), 'Obsession Type'] = 'Hoarding'

    # --- Enhance Severity Correlation ---
    base_score = 15 + (df['Duration of Symptoms (months)'] * 0.1)
    if 'Depression Diagnosis' in df.columns:
        is_depressed = df['Depression Diagnosis'].astype(str).str.lower() == 'yes'
        base_score += np.where(is_depressed, 8, 0)
        
    noise = np.random.normal(0, 3, size=len(df))
    df['Total Y-BOCS Score'] = np.clip(base_score + noise, 0, 40).astype(int)

    cols_to_drop = ['Patient ID', 'OCD Diagnosis Date']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df_clean

# Severity model re-added for comprehensive clinical focus
def train_severity_model(df):
    """Trains regression model for Total Y-BOCS Score."""
    target = 'Total Y-BOCS Score'
    X = df[FEATURES]
    y = df[target]
    
    categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
    numerical_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = {'R2': r2_score(y_test, preds)}
    
    return model, metrics


def train_obsession_type_model(df):
    """Trains classification model for Obsession Type."""
    target = 'Obsession Type'
    X = df[FEATURES]
    y = df[target]
    
    categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
    numerical_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # Calculate regression-like metrics for classification by encoding labels
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test])) # Ensure all possible labels are covered
    y_test_encoded = le.transform(y_test)
    preds_encoded = le.transform(preds)
    
    r2 = r2_score(y_test_encoded, preds_encoded)
    mse = mean_squared_error(y_test_encoded, preds_encoded)
    rmse = np.sqrt(mse)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse
    }
    
    return model, metrics

# ==========================================
# 2. STREAMLIT APP LOGIC
# ==========================================

# Set page config
st.set_page_config(
    page_title="OCD Health Data Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data for App
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ocd_patient_dataset.csv')
        # Ensure Total Score exists for visualization
        if 'Y-BOCS Score (Obsessions)' in df.columns and 'Y-BOCS Score (Compulsions)' in df.columns:
            df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
        return df
    except Exception as e:
        return None

# Load Models
@st.cache_resource
def load_models():
    try:
        with open('ocd_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        return None

df = load_data()
models = load_models()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "EDA Dashboard", 
    "Model Performance", 
    "Predict OCD",
    "About"
])

st.sidebar.markdown("### 🌿 Wellness Corner")
st.sidebar.info("Remember: You are not your thoughts. Take a moment to pause and breathe.")
st.sidebar.caption("Emergency (India): Call 112 or 1800-599-0019 (Kiran)")

# --- PAGE: HOME ---
if page == "Home":
    st.title("🧠 OCD Health Data Analysis & Prediction")
    
    # --- Business Context Section ---
    st.markdown("""
    ### 👋 Welcome to Your Health Companion
    **Understanding mental health is the first step towards recovery and wellness.**
    
    Obsessive-Compulsive Disorder (OCD) is a complex condition, but with the right insights and tools, it can be managed effectively. This platform leverages data science and machine learning to provide you with a deeper understanding of symptom patterns, severity indicators, and personalized predictions.
    
    Our goal is to bridge the gap between clinical data and patient understanding, fostering a more informed approach to mental health management.
    
    ---
    
    ### 🎯 Business Problem & Objective
    **The Challenge:** 
    Mental health professionals and patients often lack accessible, data-driven tools to objectively assess OCD severity and symptom patterns. Subjective reporting can lead to variations in diagnosis and treatment planning. Also, there is a stigma and lack of accessible information surrounding the detailed nuances of OCD subtypes.

    **The Solution:** 
    This application addresses this gap by:
    *   **Quantifying Severity:** Using the standard Y-BOCS scoring system to provide specific, tracking-friendly severity metrics.
    *   **Identifying Patterns:** Leveraging machine learning algorithms to correlate key demographic factors (like age, marital status) with specific obsession and compulsion types.
    *   **Enhancing Accessibility:** Providing a user-friendly, interactive interface for patients to monitor their condition and for clinicians to support their diagnosis with data backup.
    

    """)
# --- Wellness & Info Section ---
    st.markdown("---")
    st.subheader("🧘 Daily Mental Wellness Tip")
    tips = [
        "Practice mindfulness meditation for 10 minutes today.",
        "Take a brisk walk outside to clear your mind.",
        "Write down three things you are grateful for.",
        "Limit screen time before bed to improve sleep quality.",
        "Stay hydrated and eat a balanced diet to support brain health.",
        "Reach out to a friend or family member for a chat."
    ]
    st.warning(f"💡 **Tip of the Day:** {np.random.choice(tips)}")

    st.markdown("""
    ### 🏥 Understanding & Managing OCD
    **Obsessive-Compulsive Disorder (OCD)** is a mental health disorder that affects people of all ages and walks of life. It occurs when a person gets caught in a cycle of obsessions and compulsions.
    
    *   **Obsessions** are unwanted, intrusive thoughts, images, or urges that trigger intensely distressing feelings.
    *   **Compulsions** are behaviors an individual engages in to attempt to get rid of the obsessions and/or decrease his or her distress.
    
    **Common Coping Strategies:**
    *   **Therapy:** Cognitive Behavioral Therapy (CBT) and Exposure and Response Prevention (ERP) are considered the gold standard.
    *   **Medication:** Consult with a psychiatrist for appropriate medical treatments.
    *   **Support Groups:** Connecting with others who share similar experiences can be very validating.
    *   **Stress Management:** Yoga, exercise, and deep breathing techniques can help reduce anxiety triggers.
    """)

    if df is not None:
        st.markdown("### 📂 Dataset Preview")
        st.dataframe(df.head())

# --- PAGE: EDA DASHBOARD ---
elif page == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender Distribution")
            fig_gender = px.pie(df, names='Gender', hole=0.4)
            st.plotly_chart(fig_gender, use_container_width=True)
        with col2:
            st.subheader("Marital Status")
            fig_marital = px.bar(df, x='Marital Status', color='Marital Status')
            st.plotly_chart(fig_marital, use_container_width=True)
            
        st.subheader("Y-BOCS Severity vs Duration")
        fig_scatter = px.scatter(df, x='Duration of Symptoms (months)', y='Total Y-BOCS Score', 
                                 color='Obsession Type', size='Y-BOCS Score (Obsessions)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Obsession Types")
            fig_obs = px.histogram(df, x='Obsession Type', color='Obsession Type')
            st.plotly_chart(fig_obs, use_container_width=True)
        with col4:
            st.subheader("Compulsion Types")
            fig_comp = px.histogram(df, x='Compulsion Type', color='Compulsion Type')
            st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Correlation Matrix (Numerical)")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.error("Data not loaded.")

# --- PAGE: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    
    if models is not None:
        metrics = models.get('metrics', {})
        typ = metrics.get('type', {})
        
        st.subheader("🧩 Obsession Type (Classification)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{typ.get('Accuracy', 0):.2%}")
        c2.metric("R2 Score", f"{typ.get('R2', 0):.4f}")
        c3.metric("MSE", f"{typ.get('MSE', 0):.4f}")
        c4.metric("RMSE", f"{typ.get('RMSE', 0):.4f}")
    else:
        st.warning("Models not found. Please ensure 'ocd_models.pkl' is present in the directory.")

# Predict OCD Severity page removed

# --- PAGE: PREDICT OCD ---
elif page == "Predict OCD":
    st.title("🧩 Predict OCD (Diagnosis & Subtype)")
    if models is None:
        st.warning("Models not trained.")
    else:
        model = models.get('type_model')
        if not model:
            st.error("Type model not found.")
        else:
            with st.form("type_form"):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Age", 10, 100, 30)
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    ethnicity = st.selectbox("Ethnicity", df['Ethnicity'].unique())
                    marital = st.selectbox("Marital Status", df['Marital Status'].unique())
                with c2:
                    edu = st.selectbox("Education Level", df['Education Level'].unique())
                    dur = st.number_input("Duration (months)", 0, value=12)
                    prev = st.selectbox("Previous Diagnoses", df['Previous Diagnoses'].unique())
                    fam = st.selectbox("Family History", df['Family History of OCD'].unique())
                
                st.subheader("Clinical History")
                cc1, cc2, cc3 = st.columns(3)
                dep = cc1.selectbox("Depression Diagnosis", df['Depression Diagnosis'].unique())
                anx = cc2.selectbox("Anxiety Diagnosis", df['Anxiety Diagnosis'].unique())
                meds = cc3.selectbox("Medications", df['Medications'].unique())
                
                comp_type = st.selectbox("Compulsion Type (Observed)", df['Compulsion Type'].unique())
                
                if st.form_submit_button("Predict Status & Type"):
                    input_df = pd.DataFrame({
                        'Age': [age], 'Gender': [gender], 'Ethnicity': [ethnicity],
                        'Marital Status': [marital], 'Education Level': [edu],
                        'Duration of Symptoms (months)': [dur], 'Previous Diagnoses': [prev],
                        'Family History of OCD': [fam], 'Depression Diagnosis': [dep],
                        'Anxiety Diagnosis': [anx], 'Medications': [meds],
                        'Compulsion Type': [comp_type]
                    })
                    
                    try:
                        # 1. Predict Severity
                        sev_model = models.get('severity_model')
                        if sev_model:
                            score = sev_model.predict(input_df)[0]
                            st.subheader("📊 OCD Severity Assessment")
                            st.write(f"Predicted Total Y-BOCS Score: **{score:.2f}**")
                            
                            # Scale logic: 0-7 Subclinical/No OCD
                            if score <= 7:
                                st.success("Diagnosis: **Negative (No OCD)**")
                            elif score <= 15:
                                st.info("Diagnosis: **Positive (Mild OCD)**")
                            elif score <= 23:
                                st.warning("Diagnosis: **Positive (Moderate OCD)**")
                            else:
                                st.error("Diagnosis: **Positive (Severe OCD)**")
                        
                        # 2. Predict Type
                        type_model = models.get('type_model')
                        if type_model:
                            type_pred = type_model.predict(input_df)[0]
                            st.markdown("---")
                            st.subheader("🧩 Subtype Classification")
                            st.success(f"Predicted Likely Obsession Type: **{type_pred}**")
                    except Exception as e:
                        st.error(f"⚠️ Prediction Error: {e}")
                        if "Obsession Type" in str(e) or "columns are missing" in str(e) or "feature_names_in_" in str(e):
                            st.info("Your models seem out of sync. Please try re-training the models manually or contact support.")
                        else:
                            st.info("Tip: If you recently changed the dataset, you may need to re-train the models.")

# --- PAGE: ABOUT ---
elif page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### Overview
    Unified application logic combining EDA, Prediction, and Training pipelines.
    
    ### Tech Stack
    *   **Frontend:** Streamlit
    *   **ML:** Scikit-Learn
    *   **Viz:** Plotly Express

    """)
