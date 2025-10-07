# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="MIT College Placement Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# UTILITY: Load Artifacts Safely
# ===============================
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)
    
    model_path = os.path.join(BASE_DIR, "outputs/placement_prediction_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "outputs/scaler.pkl")
    features_path = os.path.join(BASE_DIR, "outputs/feature_columns.pkl")
    csv_path = os.path.join(BASE_DIR, "outputs/placement_cleaned.csv")
    
    # Check all files exist
    for f in [model_path, scaler_path, features_path, csv_path]:
        if not os.path.exists(f):
            st.error(f"⚠️ Missing file: {f}")
            st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    df = pd.read_csv(csv_path)
    
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
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>MIT College Placement Dashboard</h1>", 
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ===============================
# 1️⃣ HOME PAGE
# ===============================
if page == "Home":

    st.markdown(
        "<h4 style='text-align: center; color: #333;'>Quick Overview of Students & Placement Stats</h4>",
        unsafe_allow_html=True
    )

    df = pd.read_csv("outputs/placement_cleaned.csv")

    # Placement numeric mapping for overview only
    df['placement_numeric'] = df['placementstatus'].astype(str).str.lower().apply(lambda x: 1 if 'place' in x or x=='1' or x=='yes' else 0)

    # High-level metrics
    total_students = len(df)
    placed_count = df['placement_numeric'].sum()
    not_placed_count = total_students - placed_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", total_students)
    col2.metric("Placed", placed_count)
    col3.metric("Not Placed", not_placed_count)

    st.markdown("---")

    # Pie chart: Placement distribution
    pie_data = df['placement_numeric'].value_counts().rename(index={1:'Placed', 0:'Not Placed'}).reset_index()
    pie_data.columns = ['Placement', 'Count']
    fig_pie = px.pie(pie_data, names='Placement', values='Count', color='Placement',
                     color_discrete_map={'Placed':'green','Not Placed':'red'},
                     title="Overall Placement Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Quick Stats Box
    st.markdown("### Quick Stats")
    st.write(f"- Average CGPA: {df['cgpa'].mean():.2f}" if 'cgpa' in df.columns else "- Average CGPA: N/A")
    st.write(f"- Average Internship Count: {df['internships'].mean():.1f}" if 'internships' in df.columns else "- Avg Internships: N/A")
    st.write(f"- Average Projects Count: {df['projects'].mean():.1f}" if 'projects' in df.columns else "- Avg Projects: N/A")


# ===============================
# 2️⃣ EDA PAGE
# ===============================
elif page == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    # Dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Placement Distribution
    st.write("### Placement Distribution")
    fig1 = px.pie(df, names='placementstatus', title='Placed vs Not Placed', color='placementstatus')
    st.plotly_chart(fig1, use_container_width=True)

    # Numeric correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlations")
    st.plotly_chart(fig2, use_container_width=True)

    # Interactive scatter: CGPA vs Aptitude Score
    st.write("### CGPA vs Aptitude Test Score by Placement")
    fig3 = px.scatter(df, x='cgpa', y='aptitudetestscore', color='placementstatus', 
                      hover_data=df.columns, title="CGPA vs Aptitude Test Score")
    st.plotly_chart(fig3, use_container_width=True)

    # Interactive bar charts for categorical / ordinal features
    st.write("### Placement % by Categorical Features")
    cat_features = ['internships', 'projects', 'workshopscertifications', 'placementtraining', 'softskillsrating']
    for c in cat_features:
        ct = pd.crosstab(df[c], df['placementstatus'], normalize='index') * 100
        fig = px.bar(ct, barmode='stack', title=f"Placement % by {c}")
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# 3️⃣ MODEL COMPARISON PAGE
# ===============================
elif page == "Model Comparison":
    st.subheader("💻 Model Performance Comparison")
    try:
        results_df = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(results_df)
        st.write("✅ Shows accuracy, F1-score, and ROC-AUC for different models.")
    except:
        st.write("No model comparison file found. Run model evaluation first.")

# ===============================
# 4️⃣ PREDICT PAGE
# ===============================

@st.cache_data
def load_artifacts():
    model = joblib.load("outputs/placement_prediction_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    feature_cols = joblib.load("outputs/feature_columns.pkl")  # list of features
    df = pd.read_csv("outputs/placement_cleaned.csv")           # for default values
    return model, scaler, feature_cols, df

model, scaler, feature_cols, df = load_artifacts()

# ----------------------------
# Predict Function
# ----------------------------
def predict_student(input_dict):
    """
    input_dict: dict with raw feature names (like 'cgpa', 'internships', 'projects', 'extracurricularactivities', 'placementtraining')
    """
    # Create a row with all feature columns in correct order
    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0  # initialize all zeros

    # Map user input to proper feature columns
    for k, v in input_dict.items():
        if k in feature_cols:  # numeric features
            row.at[0, k] = v
        else:
            # For categorical features (one-hot), map Yes/No to the proper columns
            possible_cols = [col for col in feature_cols if col.startswith(k+'_')]
            for col in possible_cols:
                # If user selects "Yes", set the corresponding column to 1
                if str(v).strip().lower() in ['yes', '1', 'true']:
                    row.at[0, col] = 1
                else:
                    row.at[0, col] = 0

    # Apply scaling only to numeric columns that scaler expects
    numeric_cols = [f for f in scaler.feature_names_in_ if f in row.columns]
    if numeric_cols:
        row[numeric_cols] = scaler.transform(row[numeric_cols])

    # Predict
    prob = model.predict_proba(row)[:,1][0]
    label = int(model.predict(row)[0])
    return {"probability": float(prob), "predicted_label": label}



# ----------------------------
# Streamlit Predict Page
# ----------------------------
st.header("Predict Student Placement")
st.markdown(
    """
    Enter the student details below. The model will predict if the student is likely to be placed.
    """
)

# Prepare default values from median (numeric) or 0 (categorical)
defaults = {}
for col in feature_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        defaults[col] = float(df[col].median())
    else:
        defaults[col] = 0

# User input
user_input = {}
st.subheader("Student Details")
with st.form("placement_form"):
    for col in feature_cols:
        user_input[col] = st.number_input(col, value=defaults[col])
    submit_btn = st.form_submit_button("Predict Placement")

# Predict
if submit_btn:
    result = predict_student(user_input)
    st.success(f"Predicted Placement: {'Placed' if result['predicted_label']==1 else 'Not Placed'}")
    st.info(f"Probability of being placed: {result['probability']*100:.2f}%")
