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
    st.markdown(
        "<h1 style='text-align: center; color: #2E86C1; font-family:Sans-serif;'>MIT College Placement Dashboard</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size:18px; color: #34495E;'>Explore, visualize, and predict student placements using interactive tools.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Fix placement status values if needed
    # Ensure 'placementstatus' column is consistent
    df['placementstatus'] = df['placementstatus'].astype(str).str.strip().str.title()
    df['placementstatus'] = df['placementstatus'].replace({'Placed': 'Placed', 'Not Placed': 'Not Placed'})

    # Key statistics
    total_students = df.shape[0]
    placed_count = df[df['placementstatus'] == 'Placed'].shape[0]
    not_placed_count = df[df['placementstatus'] == 'Not Placed'].shape[0]
    avg_cgpa = round(df['cgpa'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Placed Students", placed_count)
    col3.metric("Not Placed Students", not_placed_count)
    col4.metric("Average CGPA", avg_cgpa)

    st.markdown("---")

    # Quick interactive filters
    st.subheader("📊 Explore Data Quickly")
    col1, col2 = st.columns(2)
    with col1:
        internships_filter = st.slider(
            "Internships Completed",
            int(df['internships'].min()), int(df['internships'].max()),
            (int(df['internships'].min()), int(df['internships'].max()))
        )
    with col2:
        projects_filter = st.slider(
            "Projects Completed",
            int(df['projects'].min()), int(df['projects'].max()),
            (int(df['projects'].min()), int(df['projects'].max()))
        )

    # Filtered dataset
    filtered_df = df[
        (df['internships'] >= internships_filter[0]) & (df['internships'] <= internships_filter[1]) &
        (df['projects'] >= projects_filter[0]) & (df['projects'] <= projects_filter[1])
    ]

    st.write(f"Showing first 10 rows of {filtered_df.shape[0]} students after filter:")
    st.dataframe(filtered_df.head(10))

    # Placement distribution chart
    st.subheader("Placement Distribution")
    placement_counts = filtered_df['placementstatus'].value_counts()
    st.bar_chart(placement_counts)


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
if page == "Predict":
    st.subheader("Predict Student Placement")
    st.write(
        "Adjust the student attributes below and click **Predict** to see the placement probability and predicted status."
    )

    import joblib

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

    # Only numeric features for sliders
    numeric_features = [f for f in feature_cols if f in X.select_dtypes(include=['int64','float64']).columns]

    st.write("### Adjust Student Attributes")
    user_input = {}

    for feature in numeric_features:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        median_val = float(X[feature].median())
        user_input[feature] = st.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=median_val
        )

    # Predict button
    if st.button("Predict Placement"):
        # Create a row with all features
        row = pd.DataFrame(columns=feature_cols)
        row.loc[0] = 0  # default all missing features to 0

        # Fill user input
        for k, v in user_input.items():
            if k in row.columns:
                row.at[0, k] = v

        # Scale if scaler exists
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
