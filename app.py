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

    st.write("### Internships vs Projects")
    if 'internships' in df.columns and 'projects' in df.columns:
        fig_scatter = px.scatter(df, x='internships', y='projects', color='placement_numeric',
                                 color_discrete_map={1:'green',0:'red'},
                                 labels={'placement_numeric':'Placement'},
                                 title='Internships vs Projects by Placement')
    st.plotly_chart(fig_scatter, use_container_width=True)

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
if page == "Predict":
    st.subheader("Predict Student Placement")
    st.write(
        "Adjust the student attributes below and click **Predict** to see the placement probability and predicted status."
    )


    # Load model, scaler, and feature columns
    @st.cache_data
    def load_artifacts():
        model = joblib.load("outputs/placement_prediction_model.pkl")
        feature_cols = joblib.load("outputs/feature_columns.pkl")
        try:
            scaler = joblib.load("outputs/scaler.pkl")
        except:
            scaler = None
        return model, feature_cols, scaler

    model, feature_cols, scaler = load_artifacts()

    # Load cleaned dataset to extract min/max/median for numeric features
    df_cleaned = pd.read_csv("outputs/placement_cleaned.csv")

    # Determine numeric features present in both cleaned dataset and model features
    numeric_features = [f for f in feature_cols if f in df_cleaned.select_dtypes(include=[np.number]).columns]

    st.write("### Adjust Student Attributes")
    user_input = {}

    # Create sliders dynamically
    for feature in numeric_features:
        min_val = float(df_cleaned[feature].min())
        max_val = float(df_cleaned[feature].max())
        median_val = float(df_cleaned[feature].median())
        user_input[feature] = st.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=median_val
        )

    # Predict button
    if st.button("Predict Placement"):
        # Create a row with all features in the correct order
        row = pd.DataFrame(columns=feature_cols)
        row.loc[0] = 0  # default all missing features to 0

        # Fill user input
        for k, v in user_input.items():
            if k in row.columns:
                row.at[0, k] = v

        # Apply scaler if exists
        if scaler:
            row_scaled = scaler.transform(row)
        else:
            row_scaled = row.values

        # Prediction
        prob = model.predict_proba(row_scaled)[:,1][0]  # probability of placement
        pred_label = int(model.predict(row_scaled)[0])

        # Display results
        st.success(f"Predicted Placement Probability: {prob*100:.2f}%")
        st.info(f"Predicted Placement Status: {'Placed' if pred_label==1 else 'Not Placed'}")

