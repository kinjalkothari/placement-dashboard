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
if page == "Predict":
    st.header("Predict Placement Probability for a Student")
    if model is None or scaler is None or feature_cols is None:
        st.error("Model scaler or feature list not found in outputs/. Make sure you ran the notebook snippet to save artifacts.")
    else:
        # Determine ranges from cleaned df if available
        if cleaned_df is not None:
            ranges = cleaned_df[feature_cols].describe().to_dict()
        else:
            ranges = {}
        # Input form
        with st.form("predict_form"):
            st.subheader("Student inputs")
            inputs = {}
            # We'll ask base inputs (not derived) and compute derived ones automatically
            # Base numerical inputs (use reasonable ranges if present in cleaned_df)
            def num_input(key, label, default=None, min_val=None, max_val=None, step=0.1):
                if default is None:
                    default = float((min_val + max_val)/2) if (min_val is not None and max_val is not None) else 0.0
                return st.number_input(label, value=float(default), min_value=float(min_val) if min_val is not None else None, max_value=float(max_val) if max_val is not None else None, step=step)

            # create inputs for the base raw features used earlier
            base_features = ['cgpa','internships','projects','workshopscertifications','aptitudetestscore','softskillsrating','extracurricularactivities','placementtraining','ssc_marks','hsc_marks']
            for f in base_features:
                if f in ranges:
                    r = ranges[f]
                    default = r['50%'] if '50%' in r else (r['mean'] if 'mean' in r else 0)
                    minv = r['min'] if 'min' in r else None
                    maxv = r['max'] if 'max' in r else None
                else:
                    default, minv, maxv = 0.0, None, None
                inputs[f] = num_input(f, f.replace('_',' ').title(), default=default, min_val=minv, max_val=maxv, step=0.1)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # compute derived features the same way you used in notebook
            overall_score = inputs['cgpa']*0.4 + inputs['aptitudetestscore']*0.3 + inputs['softskillsrating']*0.3
            academic_average = (inputs['ssc_marks'] + inputs['hsc_marks']) / 2.0
            experience_index = inputs['internships'] + inputs['projects'] + inputs['workshopscertifications']

            # build the feature vector in correct order
            x_dict = {
                'cgpa': inputs['cgpa'],
                'internships': inputs['internships'],
                'projects': inputs['projects'],
                'workshopscertifications': inputs['workshopscertifications'],
                'aptitudetestscore': inputs['aptitudetestscore'],
                'softskillsrating': inputs['softskillsrating'],
                'extracurricularactivities': inputs['extracurricularactivities'],
                'placementtraining': inputs['placementtraining'],
                'ssc_marks': inputs['ssc_marks'],
                'hsc_marks': inputs['hsc_marks'],
                'overall_score': overall_score,
                'academic_average': academic_average,
                'experience_index': experience_index
            }
            # create df row and scale
            x_row = pd.DataFrame([x_dict])[feature_cols]
            x_scaled = scaler.transform(x_row)
            prob = model.predict_proba(x_scaled)[0,1]
            pred = model.predict(x_scaled)[0]

            st.metric("Placement probability", f"{prob*100:.2f}%")
            st.write("Predicted label:", "Placed" if pred==1 else "Not Placed")
            st.progress(int(prob*100))

            # show local contribution (simple: show feature values and importance if model supports)
            st.subheader("Input values & contribution (rough)")
            contrib_df = pd.DataFrame({'Feature': list(x_row.columns), 'Value': x_row.iloc[0].values})
            st.dataframe(contrib_df.set_index('Feature'))

# -------------------------
# Explainability page
# -------------------------
if page == "Explainability":
    st.header("Explainability & Feature Importance")
    if cleaned_df is None:
        st.error("Cleaned dataset not found; cannot compute SHAP or plots.")
    else:
        # show feature importance from RandomForest if available
        # try to find a RandomForest model in outputs
        rf_candidates = [n for n in art['all_models'].keys() if 'random' in n.lower() or 'forest' in n.lower()]
        if rf_candidates:
            rf_name = rf_candidates[0]
            rf = art['all_models'][rf_name]
            try:
                # if it has feature_importances_
                fi = getattr(rf, "feature_importances_", None)
                if fi is not None and feature_cols:
                    fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': fi}).sort_values('Importance', ascending=False)
                    st.subheader(f"Feature importance (from {rf_name})")
                    st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h'), use_container_width=True)
                    st.dataframe(fi_df)
                else:
                    st.info("RandomForest found but does not expose feature_importances_ or feature columns missing.")
            except Exception as e:
                st.error(f"Error showing RF importance: {e}")
        else:
            st.info("No RandomForest model detected in outputs/ — feature importance via RF is unavailable.")

        # SHAP (if available)
        if SHAP_AVAILABLE and model is not None and cleaned_df is not None:
            st.subheader("SHAP explanation (single instance)")
            st.write("Select an index from the dataset to explain model prediction using SHAP (if model type supported).")
            idx = st.number_input("Row index to explain (0 .. n-1)", min_value=0, max_value=len(cleaned_df)-1, value=0)
            try:
                expl_df = cleaned_df[feature_cols]
                expl_row = expl_df.iloc[[idx]]
                # If model is tree-based, use TreeExplainer
                if hasattr(model, "predict_proba") and (SHAP_AVAILABLE):
                    if 'Tree' in str(type(model)) or hasattr(model, "feature_importances_"):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model, expl_df)
                    shap_values = explainer(expl_row)
                    st.subheader("SHAP values (bar)")
                    st.pyplot(shap.plots.bar(shap_values, max_display=10, show=False))
                else:
                    st.info("Model not compatible with SHAP explainer or shap not installed correctly.")
            except Exception as e:
                st.error("SHAP explanation failed: " + str(e))
        else:
            if not SHAP_AVAILABLE:
                st.info("Install shap (`pip install shap`) for advanced local explanations.")

# -------------------------
# Download page
# -------------------------
if page == "Download":
    st.header("Download artifacts & report")
    st.write("Download saved model and prepared datasets (if present in outputs/).")
    base = "outputs"
    for f in os.listdir(base) if os.path.exists(base) else []:
        st.markdown(f"- `{f}`")
    st.info("To deploy this app online, push this repository (including outputs/) to GitHub and connect to Streamlit Cloud (instructions below).")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("Built with ❤️ by you — ready to present as your DVP mini project.")
