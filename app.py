# app.py — Streamlit Placement Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Optional SHAP for explainability (nice-to-have)
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Placement Prediction Dashboard", layout="wide", page_icon="🎓")

# -------------------------
# Helpers: load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    base = "outputs"
    # model
    model_path = os.path.join(base, "placement_prediction_model.pkl")
    if os.path.exists(model_path):
        artifacts["model"] = joblib.load(model_path)
    else:
        artifacts["model"] = None

    # scaler
    scaler_path = os.path.join(base, "scaler.pkl")
    artifacts["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # feature columns
    feat_path = os.path.join(base, "feature_columns.pkl")
    artifacts["feature_columns"] = joblib.load(feat_path) if os.path.exists(feat_path) else None

    # cleaned dataframe
    csv_path = os.path.join(base, "placement_cleaned.csv")
    artifacts["df"] = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    # model comparison table (optional)
    mc_path = os.path.join(base, "model_comparison.csv")
    artifacts["model_comparison"] = pd.read_csv(mc_path) if os.path.exists(mc_path) else None

    # also load any other models in outputs/ (for optional multi-model compare)
    joblib_files = glob.glob(os.path.join(base, "*.joblib")) + glob.glob(os.path.join(base, "*.pkl"))
    models = {}
    for p in joblib_files:
        name = os.path.basename(p)
        try:
            models[name] = joblib.load(p)
        except:
            pass
    artifacts["all_models"] = models

    return artifacts

art = load_artifacts()
model = art["model"]
scaler = art["scaler"]
feature_cols = art["feature_columns"]
cleaned_df = art["df"]
model_comp = art["model_comparison"]

# -------------------------
# Page layout (sidebar)
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Comparison", "Predict", "Explainability", "Download"])

st.title("🎓 Placement Prediction — Interactive Dashboard")
st.markdown("Interactive dashboard for exploring placement dataset, comparing models, and predicting placement probability for a student.")

# -------------------------
# Home page
# -------------------------
if page == "Home":
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("Project Overview")
        st.write("""
        This dashboard visualizes the placement dataset, compares models, and lets you predict whether a student is likely to get placed.
        - Model file loaded: `{}`  
        - Scaler loaded: `{}`  
        - Feature count: `{}`  
        """.format("Yes" if model else "No",
                   "Yes" if scaler else "No",
                   len(feature_cols) if feature_cols else "N/A"))
        st.write("**Tip:** If any artifact is missing, go back to your notebook and run the 'save artifacts' snippet provided in the project notes.")

    with col2:
        if cleaned_df is not None:
            total = cleaned_df.shape[0]
            placed_pct = cleaned_df['placementstatus'].mean()
            st.metric("Total students", total)
            st.metric("Placement rate", f"{placed_pct*100:.2f}%")
            st.write(cleaned_df[['cgpa','aptitudetestscore','softskillsrating','internships']].describe())
        else:
            st.info("No cleaned dataframe found in outputs/ — EDA will be limited.")

# -------------------------
# EDA page
# -------------------------
if page == "EDA":
    st.header("Exploratory Data Analysis")
    if cleaned_df is None:
        st.error("Cleaned dataset not found (outputs/placement_cleaned.csv). Please save it from your notebook.")
    else:
        # Placement distribution
        st.subheader("Placement Distribution")
        df = cleaned_df.copy()
        df['placement_label'] = df['placementstatus'].map({1:'Placed', 0:'Not Placed'})
        fig = px.pie(df, names='placement_label', title="Placed vs Not Placed", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        # Histogram: CGPA and Aptitude
        st.subheader("CGPA & Aptitude Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df, x='cgpa', nbins=30, title="CGPA Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x='aptitudetestscore', nbins=30, title="Aptitude Test Score")
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
        st.plotly_chart(fig3, use_container_width=True)

        # Scatter: CGPA vs Aptitude coloured by placement
        st.subheader("CGPA vs Aptitude (by Placement)")
        fig4 = px.scatter(df, x='cgpa', y='aptitudetestscore', color='placement_label', hover_data=['internships','softskillsrating'])
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Model Comparison page
# -------------------------
if page == "Model Comparison":
    st.header("Model Performance Comparison")
    if model_comp is not None:
        st.dataframe(model_comp)
        st.bar_chart(model_comp.set_index('Model')[['Accuracy','ROC-AUC']])
    else:
        st.info("No precomputed comparison found (outputs/model_comparison.csv). Attempting to compute from models and cleaned data...")
        if cleaned_df is None:
            st.error("Cannot compute comparison — cleaned df missing.")
        else:
            # Try to compute quick comparison by re-splitting df
            df = cleaned_df.copy()
            feature_cols_local = feature_cols
            if not feature_cols_local:
                st.error("feature_columns.pkl missing — cannot compute.")
            else:
                X = df[feature_cols_local]
                y = df['placementstatus']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                if scaler:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test.values
                # load models found in outputs (all_models)
                models = art["all_models"]
                results = []
                for name, m in models.items():
                    # skip non-estimator objects (scaler etc)
                    if hasattr(m, "predict_proba"):
                        try:
                            y_pred = m.predict(X_test_scaled)
                            y_prob = m.predict_proba(X_test_scaled)[:,1]
                            acc = accuracy_score(y_test, y_pred)
                            roc = roc_auc_score(y_test, y_prob)
                            results.append({'Model': name, 'Accuracy': acc, 'ROC-AUC': roc})
                        except Exception as e:
                            pass
                if results:
                    rdf = pd.DataFrame(results)
                    st.dataframe(rdf)
                    st.bar_chart(rdf.set_index('Model')[['Accuracy','ROC-AUC']])
                else:
                    st.warning("No compatible models found in outputs/ to compute comparison.")

# -------------------------
# Predict page
# -------------------------
elif page == "Predict Placement":
    st.title("🎯 Predict Student Placement")
    st.write("Enter student details to predict placement chances:")

    user_input = {}
    col1, col2 = st.columns(2)

    # Create interactive input boxes dynamically
    for i, feature in enumerate(feature_cols):
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            user_input[feature] = val

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Preprocess & Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

    # Output Results
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ The student is **likely to get placed!** (Confidence: {round(proba*100,2)}%)")
    else:
        st.error(f"❌ The student is **less likely to get placed.** (Confidence: {round(proba*100,2)}%)")

# -------------------------
# Explainability page
# -------------------------
    
# ================================
# 3️⃣ MODEL EXPLAINABILITY PAGE
# ================================
elif page == "Explainability":
    st.title("🧠 Model Explainability (SHAP)")
    st.write("This section helps you understand which factors influence placement predictions most.")

    try:
        # Sample a few data points
        X_sample = df[feature_cols].sample(50, random_state=42)
        X_scaled = scaler.transform(X_sample)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)

        # Summary Plot
        st.subheader("Feature Importance (SHAP Summary)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig)

        st.write("**Interpretation:** Higher SHAP value → greater influence on placement prediction.")
    except Exception as e:
        st.error("⚠️ Explainability failed to load. Please ensure SHAP supports your model type.")
        st.exception(e)
