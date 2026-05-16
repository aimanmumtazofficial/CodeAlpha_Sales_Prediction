# ================================================================
# SALES PREDICTION DASHBOARD USING STREAMLIT
# CODEALPHA DATA SCIENCE INTERNSHIP - TASK 4
# Student: Aiman | ID: CA/DF1/54987
# ================================================================

# ================================================================
# IMPORT LIBRARIES
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================================================================
# COMPLETE DARK THEME CSS
# ================================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(to bottom right, #0b1020, #111827);
    color: white;
}

/* Main Container */
.main {
    background-color: transparent;
    color: white;
}

/* Header */
header {
    background-color: rgba(0,0,0,0) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #111827, #0f172a);
    border-right: 1px solid #374151;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Titles */
h1, h2, h3, h4, h5, h6 {
    color: #f9fafb !important;
    font-weight: 700;
}

/* Paragraphs */
p, label, div, span {
    color: #e5e7eb !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: linear-gradient(to right, #1e293b, #111827);
    border: 1px solid #374151;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
}

/* Tables */
table {
    color: white !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(to right, #9333ea, #ec4899);
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Sliders */
.stSlider label {
    color: white !important;
}

/* Radio Buttons */
.stRadio label {
    color: white !important;
}

/* Success Message */
.stSuccess {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
}

/* Info Box */
.stAlert {
    border-radius: 12px;
}

/* Remove Footer */
footer {
    visibility: hidden;
}

/* Toolbar */
[data-testid="stToolbar"] {
    right: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# DARK THEME FOR MATPLOTLIB
# ================================================================

plt.style.use("dark_background")

# ================================================================
# DASHBOARD TITLE
# ================================================================

st.title("📊 Sales Prediction Dashboard")

st.markdown("""
### Machine Learning Based Advertising Sales Analysis

This dashboard provides:
- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Model Training
- Sales Prediction
- Feature Importance
- Business Insights
""")

# ================================================================
# LOAD DATASET
# ================================================================

sales = pd.read_csv("Advertising.csv")

# ================================================================
# RENAME COLUMNS
# ================================================================

sales.columns = [
    "Index",
    "TV",
    "Radio",
    "Newspaper",
    "Sales"
]

# ================================================================
# REMOVE INDEX COLUMN
# ================================================================

sales.drop("Index", axis=1, inplace=True)

# ================================================================
# FEATURE ENGINEERING
# ================================================================

sales["Total_Advertising"] = (
    sales["TV"] +
    sales["Radio"] +
    sales["Newspaper"]
)

# ================================================================
# HANDLE OUTLIERS
# ================================================================

outlier_cols = ["TV", "Radio", "Newspaper", "Sales"]

for col in outlier_cols:

    Q1 = sales[col].quantile(0.25)
    Q3 = sales[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    sales[col] = sales[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================

st.sidebar.title("📌 Dashboard Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "Dataset Analysis",
        "Visualizations",
        "Model Training",
        "Sales Prediction",
        "Business Insights"
    ]
)

# ================================================================
# DASHBOARD OVERVIEW
# ================================================================

if section == "Dashboard Overview":

    st.header("📁 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        sales.shape[0]
    )

    col2.metric(
        "Total Columns",
        sales.shape[1]
    )

    col3.metric(
        "Average Sales",
        round(sales["Sales"].mean(), 2)
    )

    col4.metric(
        "Maximum Sales",
        round(sales["Sales"].max(), 2)
    )

    st.subheader("📄 Dataset Preview")

    st.dataframe(sales.head(10))

    st.subheader("📌 Dataset Information")

    info_df = pd.DataFrame({
        "Column Name": sales.columns,
        "Data Type": sales.dtypes.astype(str),
        "Missing Values": sales.isnull().sum().values
    })

    st.dataframe(info_df)

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        sales.describe().round(2)
    )

# ================================================================
# DATASET ANALYSIS
# ================================================================

elif section == "Dataset Analysis":

    st.header("📈 Dataset Analysis")

    st.subheader("Correlation Matrix")

    corr = sales.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="RdPu",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Top 5 Highest Sales")

    highest_sales = sales.nlargest(
        5,
        "Sales"
    )[[
        "TV",
        "Radio",
        "Newspaper",
        "Sales"
    ]]

    st.dataframe(highest_sales)

    st.subheader("Top 5 Lowest Sales")

    lowest_sales = sales.nsmallest(
        5,
        "Sales"
    )[[
        "TV",
        "Radio",
        "Newspaper",
        "Sales"
    ]]

    st.dataframe(lowest_sales)

# ================================================================
# VISUALIZATIONS
# ================================================================

elif section == "Visualizations":

    st.header("📊 Data Visualizations")

    # ============================================================
    # GRAPH 1
    # ============================================================

    st.subheader("1. Sales Distribution")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    sns.histplot(
        sales["Sales"],
        bins=20,
        kde=True,
        color="#9b59f5",
        ax=ax1
    )

    ax1.set_title("Sales Distribution")

    st.pyplot(fig1)

    # ============================================================
    # GRAPH 2
    # ============================================================

    st.subheader("2. TV Advertising vs Sales")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.regplot(
        x=sales["TV"],
        y=sales["Sales"],
        scatter_kws={'color':'#e879b0'},
        line_kws={'color':'#5bbfde'},
        ax=ax2
    )

    st.pyplot(fig2)

    # ============================================================
    # GRAPH 3
    # ============================================================

    st.subheader("3. Radio Advertising vs Sales")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    sns.regplot(
        x=sales["Radio"],
        y=sales["Sales"],
        scatter_kws={'color':'#9b59f5'},
        line_kws={'color':'#50e3c2'},
        ax=ax3
    )

    st.pyplot(fig3)

    # ============================================================
    # GRAPH 4
    # ============================================================

    st.subheader("4. Newspaper Advertising vs Sales")

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    sns.regplot(
        x=sales["Newspaper"],
        y=sales["Sales"],
        scatter_kws={'color':'#f5a623'},
        line_kws={'color':'#e879b0'},
        ax=ax4
    )

    st.pyplot(fig4)

    # ============================================================
    # GRAPH 5
    # ============================================================

    st.subheader("5. Advertising Budget Breakdown")

    budget = [
        sales["TV"].sum(),
        sales["Radio"].sum(),
        sales["Newspaper"].sum()
    ]

    labels = [
        "TV",
        "Radio",
        "Newspaper"
    ]

    fig5, ax5 = plt.subplots(figsize=(8, 8))

    ax5.pie(
        budget,
        labels=labels,
        autopct='%1.1f%%',
        colors=['#e879b0', '#9b59f5', '#5bbfde']
    )

    st.pyplot(fig5)

# ================================================================
# MODEL TRAINING
# ================================================================

elif section == "Model Training":

    st.header("🤖 Model Training & Evaluation")

    X = sales[["TV", "Radio", "Newspaper"]]

    y = sales["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ============================================================
    # SIMPLE LINEAR REGRESSION
    # ============================================================

    X_single = sales[["TV"]]

    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single,
        y,
        test_size=0.20,
        random_state=42
    )

    simple_model = LinearRegression()

    simple_model.fit(
        X_train_single,
        y_train_single
    )

    simple_predictions = simple_model.predict(
        X_test_single
    )

    r2_simple = r2_score(
        y_test_single,
        simple_predictions
    )

    # ============================================================
    # MULTIPLE LINEAR REGRESSION
    # ============================================================

    multiple_model = LinearRegression()

    multiple_model.fit(X_train, y_train)

    multiple_predictions = multiple_model.predict(X_test)

    r2_multiple = r2_score(
        y_test,
        multiple_predictions
    )

    # ============================================================
    # RANDOM FOREST REGRESSOR
    # ============================================================

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    r2_rf = r2_score(
        y_test,
        rf_predictions
    )

    # ============================================================
    # METRICS
    # ============================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Simple Linear Regression",
        f"{r2_simple:.4f}"
    )

    col2.metric(
        "Multiple Linear Regression",
        f"{r2_multiple:.4f}"
    )

    col3.metric(
        "Random Forest Regressor",
        f"{r2_rf:.4f}"
    )

    # ============================================================
    # ACTUAL VS PREDICTED
    # ============================================================

    st.subheader("Actual vs Predicted Sales")

    fig6, ax6 = plt.subplots(figsize=(8, 6))

    ax6.scatter(
        y_test,
        rf_predictions,
        color='#9b59f5',
        alpha=0.7
    )

    min_val = min(y_test.min(), rf_predictions.min())
    max_val = max(y_test.max(), rf_predictions.max())

    ax6.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        color='#e879b0'
    )

    st.pyplot(fig6)

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    st.subheader("Feature Importance")

    importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig7, ax7 = plt.subplots(figsize=(9, 5))

    ax7.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=['#e879b0', '#9b59f5', '#5bbfde']
    )

    st.pyplot(fig7)

# ================================================================
# SALES PREDICTION
# ================================================================

elif section == "Sales Prediction":

    st.header("📈 Predict Future Sales")

    st.subheader("Enter Advertising Budget")

    tv = st.slider(
        "TV Advertising Budget",
        0,
        300,
        150
    )

    radio = st.slider(
        "Radio Advertising Budget",
        0,
        50,
        25
    )

    newspaper = st.slider(
        "Newspaper Advertising Budget",
        0,
        120,
        40
    )

    X = sales[["TV", "Radio", "Newspaper"]]

    y = sales["Sales"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    prediction_input = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    predicted_sales = model.predict(
        prediction_input
    )[0]

    st.success(
        f"Predicted Sales: {predicted_sales:.2f}"
    )

# ================================================================
# BUSINESS INSIGHTS
# ================================================================

elif section == "Business Insights":

    st.header("💡 Business Insights")

    st.markdown("""
    ## Key Findings

    ### 📺 TV Advertising
    - TV advertising has the strongest impact on sales.
    - Higher TV spending produces better customer reach.
    - Businesses investing heavily in TV achieve stronger sales growth.

    ### 📻 Radio Advertising
    - Radio advertising has a positive effect on sales.
    - Radio campaigns improve customer engagement.
    - Combining TV and Radio creates better marketing performance.

    ### 📰 Newspaper Advertising
    - Newspaper advertising contributes less compared to TV and Radio.
    - Newspaper campaigns should be used as supporting marketing channels.

    ### 🤖 Machine Learning Insights
    - Random Forest Regressor achieved the best prediction accuracy.
    - Machine learning helps forecast future sales effectively.
    - Predictive analytics improves business decision making.

    ### 📈 Marketing Recommendations
    - Increase investment in TV advertising.
    - Use Radio advertising for stronger engagement.
    - Optimize advertising budgets using data analysis.
    - Use predictive models before launching campaigns.
    - Track campaign performance regularly.

    ### 🚀 Business Growth Strategy
    - Data-driven advertising increases ROI.
    - Balanced advertising channels improve performance.
    - Smart budget allocation reduces unnecessary expenses.
    - Companies can forecast sales before campaign execution.
    """)

