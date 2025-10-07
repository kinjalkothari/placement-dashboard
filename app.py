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
    page_title="MIT College Placement Dashboard",
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
    st.subheader("Welcome to MIT College Placement Dashboard")
    
    # Hero Image
    st.image(
        "https://images.unsplash.com/photo-1596496053374-3f04f0d0d9d0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8Y29sbGVnZXxlbnwwfHwwfHw%3D&ixlib=rb-4.0.3&q=80&w=1080",
        use_container_width=True
    )

    st.write(
        """
        This interactive dashboard allows you to:
        - Explore placement data of MIT College students
        - Visualize trends and correlations
        - Compare machine learning models predicting placement
        - Predict the likelihood of a student getting placed based on their attributes
        """
    )

    st.markdown("---")

    # Key statistics
    total_students = df.shape[0]
    placed_count = df[df['placementstatus']=='Placed'].shape[0]
    not_placed_count = df[df['placementstatus']=='Not Placed'].shape[0]
    avg_cgpa = round(df['cgpa'].mean(), 2)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Placed Students", placed_count)
    col3.metric("Not Placed Students", not_placed_count)
    col4.metric("Average CGPA", avg_cgpa)

    st.markdown("---")

    # Quick filter options for visual exploration
    st.write("### Quick Filters")
    col1, col2 = st.columns(2)
    with col1:
        internship_filter = st.slider("Internships Completed", int(df['internships'].min()), int(df['internships'].max()), (int(df['internships'].min()), int(df['internships'].max())))
    with col2:
        project_filter = st.slider("Projects Completed", int(df['projects'].min()), int(df['projects'].max()), (int(df['projects'].min()), int(df['projects'].max())))

    filtered_df = df[(df['internships'] >= internship_filter[0]) & (df['internships'] <= internship_filter[1]) &
                     (df['projects'] >= project_filter[0]) & (df['projects'] <= project_filter[1])]

    st.write(f"Filtered Dataset Preview ({filtered_df.shape[0]} students)")
    st.dataframe(filtered_df.head())


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
elif page == "Predict":
    st.subheader("🎯 Predict Student Placement")
    st.write("Enter student details below to predict placement:")

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
