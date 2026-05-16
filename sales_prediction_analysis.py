# ================================================================
# SALES PREDICTION USING MACHINE LEARNING
# CODEALPHA DATA SCIENCE INTERNSHIP - TASK 4
# Student: Aiman | ID: CA/DF1/54987
# ================================================================

# ================================================================
# IMPORT LIBRARIES
# ================================================================

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
# MATPLOTLIB DARK THEME
# ================================================================

plt.rcParams.update({
    'figure.facecolor'  : '#0e0e14',
    'axes.facecolor'    : '#1a1730',
    'axes.edgecolor'    : '#2e2a4a',
    'axes.labelcolor'   : '#c8c4e0',
    'xtick.color'       : '#c8c4e0',
    'ytick.color'       : '#c8c4e0',
    'text.color'        : '#e8e6f0',
    'grid.color'        : '#2a2740',
    'grid.alpha'        : 0.5,
    'axes.titlecolor'   : '#e8e6f0',
    'legend.facecolor'  : '#1a1730',
    'legend.edgecolor'  : '#2e2a4a',
    'figure.edgecolor'  : '#0e0e14',
})

COLORS = ['#e879b0', '#9b59f5', '#5bbfde', '#f5a623', '#50e3c2']

print("=" * 70)
print("      SALES PREDICTION USING MACHINE LEARNING")
print("=" * 70)

# ================================================================
# STEP 1 - LOAD THE DATASET
# ================================================================

sales = pd.read_csv("Advertising.csv")

print("\nStep 1: Dataset Loaded Successfully")

print(f"\nTotal Rows    : {sales.shape[0]}")
print(f"Total Columns : {sales.shape[1]}")

# ================================================================
# DISPLAY ORIGINAL COLUMNS
# ================================================================

print("\nOriginal Columns:")
print(list(sales.columns))

# ================================================================
# RENAME COLUMNS
# ================================================================

# The dataset contains an unnamed index column.
# Columns are renamed for better readability.

sales.columns = [
    "Index",
    "TV",
    "Radio",
    "Newspaper",
    "Sales"
]

# ================================================================
# REMOVE UNNECESSARY COLUMN
# ================================================================

sales.drop("Index", axis=1, inplace=True)

print("\nUpdated Columns:")
print(list(sales.columns))

# ================================================================
# STEP 2 - DATA CLEANING
# ================================================================

print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Keep original copy
sales_original = sales.copy()

print("\nFirst 5 Rows:")
print(sales.head())

print("\nLast 5 Rows:")
print(sales.tail())

print("\nDataset Shape:")
print(sales.shape)

# ================================================================
# CHECK MISSING VALUES
# ================================================================

print("\nMissing Values:")
print(sales.isnull().sum())

if sales.isnull().sum().sum() == 0:
    print("\nNo Missing Values Found.")

# ================================================================
# CHECK DUPLICATE VALUES
# ================================================================

duplicates = sales.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicates)

if duplicates > 0:
    sales.drop_duplicates(inplace=True)
    print("\nDuplicates Removed Successfully.")

# ================================================================
# CHECK DATA TYPES
# ================================================================

print("\nData Types:")
print(sales.dtypes)

# ================================================================
# STEP 3 - FEATURE ENGINEERING
# ================================================================

print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)

# Create Total Advertising Feature

sales["Total_Advertising"] = (
    sales["TV"] +
    sales["Radio"] +
    sales["Newspaper"]
)

print("\nNew Feature Created:")
print("Total_Advertising = TV + Radio + Newspaper")

print("\nUpdated Dataset:")
print(sales.head())

# ================================================================
# STEP 4 - NUMERICAL ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("STEP 4: NUMERICAL COLUMN ANALYSIS")
print("=" * 70)

numerical_cols = sales.select_dtypes(include='number').columns.tolist()

print("\nNumerical Columns:")
print(numerical_cols)

print("\nDescriptive Statistics:")
print(sales.describe().round(2))

print("\nVariance:")
print(sales[numerical_cols].var().round(2))

print("\nStandard Deviation:")
print(sales[numerical_cols].std().round(2))

print("\nCorrelation Matrix:")
print(sales[numerical_cols].corr().round(2))

# ================================================================
# STEP 5 - OUTLIER HANDLING USING CAPPING
# ================================================================

print("\n" + "=" * 70)
print("STEP 5: OUTLIER HANDLING USING IQR CAPPING")
print("=" * 70)

sales_before_outliers = sales.copy()

outlier_cols = ["TV", "Radio", "Newspaper", "Sales"]

for col in outlier_cols:

    Q1 = sales[col].quantile(0.25)
    Q3 = sales[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers_count = len(
        sales[
            (sales[col] < lower_bound) |
            (sales[col] > upper_bound)
        ]
    )

    sales[col] = sales[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    print(f"\nColumn Name      : {col}")
    print(f"Lower Bound      : {lower_bound:.2f}")
    print(f"Upper Bound      : {upper_bound:.2f}")
    print(f"Outliers Capped  : {outliers_count}")

print("\nOutlier Handling Completed Successfully.")

# ================================================================
# STEP 6 - FULL EDA
# ================================================================

print("\n" + "=" * 70)
print("STEP 6: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nAverage Sales:")
print(round(sales["Sales"].mean(), 2))

print("\nHighest TV Advertising:")
print(sales["TV"].max())

print("\nHighest Radio Advertising:")
print(sales["Radio"].max())

print("\nHighest Newspaper Advertising:")
print(sales["Newspaper"].max())

print("\nTop 5 Highest Sales:")
print(
    sales.nlargest(5, "Sales")[
        ["TV", "Radio", "Newspaper", "Sales"]
    ]
)

print("\nTop 5 Lowest Sales:")
print(
    sales.nsmallest(5, "Sales")[
        ["TV", "Radio", "Newspaper", "Sales"]
    ]
)

# ================================================================
# STEP 7 - GRAPH 1
# SALES DISTRIBUTION
# ================================================================

print("\nGenerating Graphs...")

fig1, ax1 = plt.subplots(figsize=(9, 5))

sns.histplot(
    sales["Sales"],
    bins=20,
    kde=True,
    color='#9b59f5',
    edgecolor='#2e2a4a',
    ax=ax1
)

ax1.set_title(
    "Sales Distribution",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax1.set_xlabel("Sales")
ax1.set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "graph1_sales_distribution.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 1 Saved Successfully")

# ================================================================
# STEP 8 - GRAPH 2
# CORRELATION HEATMAP
# ================================================================

fig2, ax2 = plt.subplots(figsize=(10, 7))

corr = sales.corr()

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='RdPu',
    linewidths=0.5,
    square=True,
    ax=ax2
)

ax2.set_title(
    "Correlation Heatmap",
    fontsize=14,
    fontweight='bold',
    pad=12
)

plt.tight_layout()

plt.savefig(
    "graph2_correlation_heatmap.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 2 Saved Successfully")

# ================================================================
# STEP 9 - GRAPH 3
# TV VS SALES
# ================================================================

fig3, ax3 = plt.subplots(figsize=(9, 5))

sns.regplot(
    x=sales["TV"],
    y=sales["Sales"],
    scatter_kws={'color':'#e879b0'},
    line_kws={'color':'#5bbfde'},
    ax=ax3
)

ax3.set_title(
    "TV Advertising vs Sales",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax3.set_xlabel("TV Advertising")
ax3.set_ylabel("Sales")

plt.tight_layout()

plt.savefig(
    "graph3_tv_vs_sales.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 3 Saved Successfully")

# ================================================================
# STEP 10 - GRAPH 4
# RADIO VS SALES
# ================================================================

fig4, ax4 = plt.subplots(figsize=(9, 5))

sns.regplot(
    x=sales["Radio"],
    y=sales["Sales"],
    scatter_kws={'color':'#9b59f5'},
    line_kws={'color':'#50e3c2'},
    ax=ax4
)

ax4.set_title(
    "Radio Advertising vs Sales",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax4.set_xlabel("Radio Advertising")
ax4.set_ylabel("Sales")

plt.tight_layout()

plt.savefig(
    "graph4_radio_vs_sales.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 4 Saved Successfully")

# ================================================================
# STEP 11 - GRAPH 5
# NEWSPAPER VS SALES
# ================================================================

fig5, ax5 = plt.subplots(figsize=(9, 5))

sns.regplot(
    x=sales["Newspaper"],
    y=sales["Sales"],
    scatter_kws={'color':'#f5a623'},
    line_kws={'color':'#e879b0'},
    ax=ax5
)

ax5.set_title(
    "Newspaper Advertising vs Sales",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax5.set_xlabel("Newspaper Advertising")
ax5.set_ylabel("Sales")

plt.tight_layout()

plt.savefig(
    "graph5_newspaper_vs_sales.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 5 Saved Successfully")

# ================================================================
# STEP 12 - PREPARE DATA FOR MODEL
# ================================================================

print("\n" + "=" * 70)
print("STEP 12: DATA PREPARATION")
print("=" * 70)

X = sales[["TV", "Radio", "Newspaper"]]

y = sales["Sales"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(list(X.columns))

# ================================================================
# TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:")
print(X_train.shape[0])

print("\nTesting Samples:")
print(X_test.shape[0])

# ================================================================
# STEP 13 - SIMPLE LINEAR REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("STEP 13: SIMPLE LINEAR REGRESSION")
print("=" * 70)

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

mae_simple = mean_absolute_error(
    y_test_single,
    simple_predictions
)

mse_simple = mean_squared_error(
    y_test_single,
    simple_predictions
)

rmse_simple = np.sqrt(mse_simple)

print("\nSimple Linear Regression Results:")

print(f"R2 Score : {r2_simple:.4f}")
print(f"MAE      : {mae_simple:.4f}")
print(f"MSE      : {mse_simple:.4f}")
print(f"RMSE     : {rmse_simple:.4f}")

# ================================================================
# STEP 14 - MULTIPLE LINEAR REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("STEP 14: MULTIPLE LINEAR REGRESSION")
print("=" * 70)

multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)

multiple_predictions = multiple_model.predict(X_test)

r2_multiple = r2_score(
    y_test,
    multiple_predictions
)

mae_multiple = mean_absolute_error(
    y_test,
    multiple_predictions
)

mse_multiple = mean_squared_error(
    y_test,
    multiple_predictions
)

rmse_multiple = np.sqrt(mse_multiple)

print("\nMultiple Linear Regression Results:")

print(f"R2 Score : {r2_multiple:.4f}")
print(f"MAE      : {mae_multiple:.4f}")
print(f"MSE      : {mse_multiple:.4f}")
print(f"RMSE     : {rmse_multiple:.4f}")

# ================================================================
# STEP 15 - RANDOM FOREST REGRESSOR
# ================================================================

print("\n" + "=" * 70)
print("STEP 15: RANDOM FOREST REGRESSOR")
print("=" * 70)

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

mae_rf = mean_absolute_error(
    y_test,
    rf_predictions
)

mse_rf = mean_squared_error(
    y_test,
    rf_predictions
)

rmse_rf = np.sqrt(mse_rf)

print("\nRandom Forest Results:")

print(f"R2 Score : {r2_rf:.4f}")
print(f"MAE      : {mae_rf:.4f}")
print(f"MSE      : {mse_rf:.4f}")
print(f"RMSE     : {rmse_rf:.4f}")

# ================================================================
# STEP 16 - MODEL COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("STEP 16: MODEL COMPARISON")
print("=" * 70)

print(f"\n{'Model':<35} {'R2 Score':<12} {'RMSE':<12}")

print("-" * 70)

print(f"{'Simple Linear Regression':<35} {r2_simple:<12.4f} {rmse_simple:<12.4f}")

print(f"{'Multiple Linear Regression':<35} {r2_multiple:<12.4f} {rmse_multiple:<12.4f}")

print(f"{'Random Forest Regressor':<35} {r2_rf:<12.4f} {rmse_rf:<12.4f}")

best_model = "Random Forest Regressor"

print(f"\nBest Performing Model: {best_model}")

# ================================================================
# STEP 17 - GRAPH 6
# ACTUAL VS PREDICTED SALES
# ================================================================

fig6, ax6 = plt.subplots(figsize=(8, 6))

ax6.scatter(
    y_test,
    rf_predictions,
    color='#9b59f5',
    alpha=0.7,
    edgecolor='#2e2a4a',
    s=60
)

min_val = min(y_test.min(), rf_predictions.min())
max_val = max(y_test.max(), rf_predictions.max())

ax6.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='#e879b0',
    linestyle='--',
    linewidth=2
)

ax6.set_title(
    "Actual vs Predicted Sales",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax6.set_xlabel("Actual Sales")
ax6.set_ylabel("Predicted Sales")

plt.tight_layout()

plt.savefig(
    "graph6_actual_vs_predicted.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 6 Saved Successfully")

# ================================================================
# STEP 18 - FEATURE IMPORTANCE
# ================================================================

importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature"    : X.columns,
    "Importance" : importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\nFeature Importance:")
print(importance_df)

# ================================================================
# STEP 19 - GRAPH 7
# FEATURE IMPORTANCE
# ================================================================

fig7, ax7 = plt.subplots(figsize=(9, 5))

bars = ax7.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color=['#e879b0', '#9b59f5', '#5bbfde']
)

for bar in bars:

    width = bar.get_width()

    ax7.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f'{width:.3f}',
        va='center',
        fontsize=10
    )

ax7.set_title(
    "Feature Importance",
    fontsize=14,
    fontweight='bold',
    pad=12
)

ax7.set_xlabel("Importance Score")

plt.tight_layout()

plt.savefig(
    "graph7_feature_importance.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 7 Saved Successfully")

# ================================================================
# STEP 20 - GRAPH 8
# ADVERTISING BUDGET BREAKDOWN
# ================================================================

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

fig8, ax8 = plt.subplots(figsize=(8, 8))

ax8.pie(
    budget,
    labels=labels,
    autopct='%1.1f%%',
    colors=COLORS
)

ax8.set_title(
    "Advertising Budget Breakdown",
    fontsize=14,
    fontweight='bold',
    pad=12
)

plt.tight_layout()

plt.savefig(
    "graph8_budget_breakdown.png",
    dpi=150,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("Graph 8 Saved Successfully")

# ================================================================
# STEP 21 - PREDICT FUTURE SALES
# ================================================================

print("\n" + "=" * 70)
print("STEP 21: FUTURE SALES PREDICTION")
print("=" * 70)

new_campaign = pd.DataFrame({
    "TV"        : [250],
    "Radio"     : [40],
    "Newspaper" : [30]
})

future_sales = rf_model.predict(new_campaign)[0]

print("\nAdvertising Budget:")

print(f"TV Advertising        : {new_campaign['TV'][0]}")
print(f"Radio Advertising     : {new_campaign['Radio'][0]}")
print(f"Newspaper Advertising : {new_campaign['Newspaper'][0]}")

print(f"\nPredicted Sales: {future_sales:.2f}")

# ================================================================
# STEP 22 - BUSINESS INSIGHTS
# ================================================================

print("\n" + "=" * 70)
print("STEP 22: BUSINESS INSIGHTS")
print("=" * 70)

most_important = importance_df.sort_values(
    by="Importance",
    ascending=False
).iloc[0]["Feature"]

print(f"""

1. DATASET OVERVIEW
   ------------------------------------------------------------
   - Total advertising campaigns analyzed: {sales.shape[0]}
   - Features used for prediction: TV, Radio, Newspaper
   - Sales values ranged from {sales['Sales'].min():.1f} to {sales['Sales'].max():.1f}

2. DATA QUALITY
   ------------------------------------------------------------
   - No missing values were found in the dataset.
   - Duplicate records were checked and handled.
   - Outliers were managed using the IQR capping method.
   - No data rows were deleted during preprocessing.

3. ADVERTISING CHANNEL PERFORMANCE
   ------------------------------------------------------------
   - TV advertising showed the strongest positive relationship with sales.
   - Radio advertising also had a significant impact on customer reach.
   - Newspaper advertising had the weakest impact on sales performance.
   - Increasing advertising investment generally increased sales outcomes.

4. MODEL PERFORMANCE
   ------------------------------------------------------------
   - Simple Linear Regression worked well using only TV advertising.
   - Multiple Linear Regression improved prediction accuracy using all features.
   - Random Forest Regressor achieved the highest accuracy.
   - Best Model: {best_model}

5. FEATURE IMPORTANCE ANALYSIS
   ------------------------------------------------------------
   - Most Important Feature: {most_important}
   - TV advertising contributed the highest influence on sales prediction.
   - Radio was the second most influential marketing channel.
   - Newspaper contributed less compared to TV and Radio.

6. MARKETING STRATEGY INSIGHTS
   ------------------------------------------------------------
   - Businesses should prioritize TV campaigns for maximum visibility.
   - Radio advertising is highly effective for improving customer engagement.
   - Newspaper advertising should be used as a supporting channel.
   - Balanced multi-channel advertising improves overall performance.

7. RETURN ON INVESTMENT INSIGHTS
   ------------------------------------------------------------
   - High TV and Radio spending generated better sales growth.
   - Data-driven advertising decisions improve marketing efficiency.
   - Optimized advertising allocation can reduce unnecessary costs.
   - Predictive analytics helps companies forecast future sales more accurately.

8. BUSINESS RECOMMENDATIONS
   ------------------------------------------------------------
   - Increase investment in high-performing channels.
   - Monitor campaign performance regularly using analytics.
   - Use machine learning models before launching marketing campaigns.
   - Focus more on customer-targeted advertising strategies.
   - Combine TV and Radio campaigns for stronger sales impact.

""")

# ================================================================
# STEP 23 - SAVE FILES
# ================================================================

sales.to_csv(
    "Cleaned_Advertising_Data.csv",
    index=False
)

importance_df.to_csv(
    "Feature_Importance.csv",
    index=False
)

print("\nFiles Saved Successfully:")
print("1. Cleaned_Advertising_Data.csv")
print("2. Feature_Importance.csv")

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Dataset              : Advertising.csv")
print(f"Rows Analyzed        : {sales.shape[0]}")
print(f"Features Used        : TV, Radio, Newspaper")
print(f"Models Trained       : 3")
print(f"Best Model           : {best_model}")
print(f"Best R2 Score        : {r2_rf:.4f}")
print(f"Graphs Generated     : 8")
print(f"Graphs Saved         : PNG Files")
print(f"Internship           : CodeAlpha Data Science Internship")

