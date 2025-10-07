import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(__file__)  # path to app.py
model_path = os.path.join(BASE_DIR, "outputs/placement_prediction_model.pkl")
scaler_path = os.path.join(BASE_DIR, "outputs/scaler.pkl")
features_path = os.path.join(BASE_DIR, "outputs/feature_columns.pkl")
csv_path = os.path.join(BASE_DIR, "outputs/placement_cleaned.csv")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(features_path)
df = pd.read_csv(csv_path)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="MIT College Placement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# LOAD ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/placement_prediction_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    feature_cols = joblib.load("outputs/feature_columns.pkl")
    df = pd.read_csv("outputs/placement_cleaned.csv")
    return model, scaler, feature_cols, df

model, scaler, feature_cols, df = load_artifacts()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Comparison", "Predict"])

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>MIT College Placement Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ===============================
# 1️⃣ HOME PAGE
# ===============================
if page == "Home":
    st.subheader("Welcome!")
    st.write("This dashboard helps visualize placement data and predict student placement based on attributes.")
    st.image("https://images.unsplash.com/photo-1569248326187-3f2e136f31b1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8Y29sbGVnZXxlbnwwfHwwfHw%3D&ixlib=rb-4.0.3&q=80&w=1080", use_column_width=True)

# ===============================
# 2️⃣ EDA PAGE
# ===============================
elif page == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Preview")
        st.dataframe(df.head())

    with col2:
        st.write("Placement Distribution")
        fig1 = px.pie(df, names='placementstatus', title='Placed vs Not Placed', color='placementstatus')
        st.plotly_chart(fig1, use_container_width=True)

    st.write("**Correlation Heatmap**")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlations")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**CGPA vs Aptitude Test Score by Placement**")
    fig3 = px.scatter(df, x='cgpa', y='aptitudetestscore', color='placementstatus', title="CGPA vs Aptitude Test Score")
    st.plotly_chart(fig3, use_container_width=True)

# ===============================
# 3️⃣ MODEL COMPARISON PAGE
# ===============================
elif page == "Model Comparison":
    st.subheader("💻 Model Performance Comparison")
    try:
        results_df = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(results_df)
        st.write("✅ This shows accuracy, F1-score, and ROC-AUC for different models tested.")
    except:
        st.write("No model comparison file found.")

# ===============================
# 4️⃣ PREDICT PAGE
# ===============================
elif page == "Predict":
    st.subheader("🎯 Predict Student Placement")
    st.write("Enter student details to predict placement chances:")

    user_input = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_cols):
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(
                label=f"{feature.replace('_', ' ').title()}",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].median())
            )
            user_input[feature] = val

    input_df = pd.DataFrame([user_input])

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Likely to get placed! (Confidence: {round(proba*100,2)}%)")
    else:
        st.error(f"❌ Less likely to get placed. (Confidence: {round((1-proba)*100,2)}%)")
