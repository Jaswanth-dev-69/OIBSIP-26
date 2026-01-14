"""
House Price Prediction using Linear Regression
Oasis Infobyte Data Analytics Internship - Level 2
Project: Predicting House Prices with Linear Regression

Author: [Your Name]
Date: January 14, 2026
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 70)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("=" * 70)
print()

# ============================================================================
# 1️⃣ DATA LOADING & EXPLORATION
# ============================================================================

print("STEP 1: DATA LOADING & EXPLORATION")
print("-" * 70)

# Load the dataset
# Note: Ensure 'housing.csv' is placed in the '../data/' directory
try:
    df = pd.read_csv('../data/housing.csv')
    print("✓ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: 'housing.csv' not found in '../data/' directory")
    print("Please download the dataset and place it in the data/ folder")
    exit()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# Display last few rows
print("Last 5 rows of the dataset:")
print(df.tail())
print()

# Dataset shape
print(f"Dataset Shape: {df.shape}")
print(f"  → {df.shape[0]} rows (samples)")
print(f"  → {df.shape[1]} columns (features)")
print()

# Column names
print("Column Names:")
print(df.columns.tolist())
print()

# Dataset information
print("Dataset Information:")
print(df.info())
print()

# Statistical summary
print("Statistical Summary:")
print(df.describe())
print()

# ============================================================================
# 2️⃣ DATA CLEANING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("-" * 70)

# Check for missing values
print("Missing Values Count:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print(f"\nTotal missing values: {missing_values.sum()}")
    
    # Handle missing values
    # Strategy: Remove rows with missing values for simplicity
    # Alternative: Fill with mean/median for numerical columns
    initial_rows = df.shape[0]
    df = df.dropna()
    rows_removed = initial_rows - df.shape[0]
    print(f"✓ Removed {rows_removed} rows with missing values")
    print(f"✓ Remaining rows: {df.shape[0]}")

print()

# Check data types
print("Data Types:")
print(df.dtypes)
print()

# Ensure all feature columns are numeric
# (Excluding non-numeric columns if any exist)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"✓ Identified {len(numeric_columns)} numeric columns")
print()

# Remove duplicates if any
initial_rows = df.shape[0]
df = df.drop_duplicates()
duplicates_removed = initial_rows - df.shape[0]
if duplicates_removed > 0:
    print(f"✓ Removed {duplicates_removed} duplicate rows")
else:
    print("✓ No duplicate rows found")
print()

print(f"✓ Final cleaned dataset shape: {df.shape}")
print()

# ============================================================================
# 3️⃣ FEATURE SELECTION
# ============================================================================

print("=" * 70)
print("STEP 3: FEATURE SELECTION")
print("-" * 70)

# Identify target variable (commonly named 'price', 'SalePrice', 'medv', etc.)
# Adjust this based on your actual dataset
possible_target_names = ['price', 'Price', 'SalePrice', 'MEDV', 'medv', 'target']
target_column = None

for col in possible_target_names:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # If standard names not found, use the last column as target
    print("⚠ Standard price column not found. Using last column as target.")
    target_column = df.columns[-1]

print(f"Target Variable: {target_column}")
print()

# Separate features (X) and target (y)
# Independent variables: All numeric columns except the target
X = df.drop(columns=[target_column])
y = df[target_column]

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

print(f"Number of Features: {X.shape[1]}")
print(f"Feature Names: {X.columns.tolist()}")
print()

print("Feature Relevance:")
print("  → Selected all numerical features that may correlate with house prices")
print("  → Features like area, rooms, location factors typically influence price")
print("  → Linear regression will determine the weight of each feature")
print()

# Display correlation with target variable
print("Correlation with Target Variable:")
correlations = df[numeric_columns].corr()[target_column].sort_values(ascending=False)
print(correlations)
print()

# ============================================================================
# 4️⃣ TRAIN-TEST SPLIT
# ============================================================================

print("=" * 70)
print("STEP 4: TRAIN-TEST SPLIT")
print("-" * 70)

# Split the data: 80% training, 20% testing
test_size = 0.2
random_state = 42  # For reproducibility

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=random_state
)

print(f"Test Size: {test_size * 100}%")
print(f"Random State: {random_state} (ensures reproducibility)")
print()

print(f"Training Set Size: {X_train.shape[0]} samples")
print(f"Testing Set Size: {X_test.shape[0]} samples")
print()

print("Why Train-Test Split is Required:")
print("  → Training set: Used to train the model and learn patterns")
print("  → Testing set: Used to evaluate model performance on unseen data")
print("  → Prevents overfitting and provides realistic performance metrics")
print("  → Ensures the model generalizes well to new data")
print()

# ============================================================================
# 5️⃣ MODEL TRAINING
# ============================================================================

print("=" * 70)
print("STEP 5: MODEL TRAINING")
print("-" * 70)

# Create Linear Regression model
model = LinearRegression()

# Train the model
print("Training Linear Regression model...")
model.fit(X_train, y_train)
print("✓ Model training completed!")
print()

# Display model parameters
print("Model Parameters:")
print(f"  → Intercept (β₀): {model.intercept_:.2f}")
print(f"  → Number of Coefficients: {len(model.coef_)}")
print()

print("Feature Coefficients (β values):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  → {feature}: {coef:.4f}")
print()

# ============================================================================
# 6️⃣ MODEL EVALUATION
# ============================================================================

print("=" * 70)
print("STEP 6: MODEL EVALUATION")
print("-" * 70)

# Make predictions on test data
print("Making predictions on test data...")
y_pred = model.predict(X_test)
print("✓ Predictions completed!")
print()

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("MODEL PERFORMANCE METRICS")
print("-" * 70)
print(f"Mean Squared Error (MSE):  {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R²) Score:      {r2:.4f}")
print()

# Interpret R² score
print("R² Score Interpretation:")
if r2 >= 0.9:
    print(f"  → {r2:.2%} - Excellent! Model explains data very well")
elif r2 >= 0.7:
    print(f"  → {r2:.2%} - Good! Model has strong predictive power")
elif r2 >= 0.5:
    print(f"  → {r2:.2%} - Moderate. Model captures some patterns")
else:
    print(f"  → {r2:.2%} - Poor. Model needs improvement")
print()

# Display sample predictions
print("Sample Predictions (First 10):")
comparison_df = pd.DataFrame({
    'Actual Price': y_test.values[:10],
    'Predicted Price': y_pred[:10],
    'Difference': y_test.values[:10] - y_pred[:10]
})
print(comparison_df)
print()

# ============================================================================
# 7️⃣ VISUALIZATION
# ============================================================================

print("=" * 70)
print("STEP 7: CREATING VISUALIZATIONS")
print("-" * 70)

# Visualization 1: Feature vs Price (using the most correlated feature)
# Select the feature with highest correlation to target
most_correlated_feature = correlations.drop(target_column).abs().idxmax()
print(f"Creating scatter plot: {most_correlated_feature} vs {target_column}...")

plt.figure(figsize=(10, 6))
plt.scatter(df[most_correlated_feature], df[target_column], alpha=0.5, color='blue')
plt.xlabel(most_correlated_feature, fontsize=12)
plt.ylabel(target_column, fontsize=12)
plt.title(f'{most_correlated_feature} vs {target_column}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('../visuals/feature_vs_price.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visuals/feature_vs_price.png")
plt.close()

# Visualization 2: Actual vs Predicted Prices
print(f"Creating scatter plot: Actual vs Predicted Prices...")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predictions')

# Add reference line (perfect prediction line)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('../visuals/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visuals/actual_vs_predicted.png")
plt.close()

print()
print("✓ All visualizations created and saved successfully!")
print()

# ============================================================================
# 📌 INSIGHTS & INTERPRETATION
# ============================================================================

print("=" * 70)
print("PROJECT INSIGHTS & INTERPRETATION")
print("=" * 70)
print()

insights = """
MODEL PERFORMANCE SUMMARY:
--------------------------
The Linear Regression model was successfully trained to predict house prices
using numerical features from the housing dataset.

R² SCORE MEANING:
-----------------
The R² (R-squared) score of {r2:.4f} ({r2_percent:.2f}%) indicates that {r2_percent:.2f}%
of the variance in house prices can be explained by the selected features.

• R² = 1.0 would mean perfect predictions
• R² = 0.0 would mean the model performs no better than predicting the mean
• Our R² = {r2:.4f} shows {interpretation}

MEAN SQUARED ERROR (MSE):
-------------------------
MSE of {mse:,.2f} represents the average squared difference between actual
and predicted prices. Lower MSE indicates better model performance.

Root Mean Squared Error (RMSE) of {rmse:,.2f} gives the average prediction
error in the same units as the target variable (price).

LIMITATIONS OF LINEAR REGRESSION:
----------------------------------
1. Assumes linear relationship between features and target
   → Real-world house prices may have non-linear patterns
   
2. Sensitive to outliers
   → Extreme values can significantly affect the model
   
3. Assumes independence of features
   → Housing features often correlate with each other (multicollinearity)
   
4. Cannot capture complex interactions
   → Real estate prices depend on many interacting factors
   
5. Assumes constant variance (homoscedasticity)
   → Price variance may differ across price ranges

POSSIBLE IMPROVEMENTS (Future Work):
-------------------------------------
1. Feature Engineering:
   • Create interaction features (e.g., size × location)
   • Add polynomial features for non-linear relationships
   • Encode categorical variables (neighborhood, style, etc.)

2. Advanced Models:
   • Ridge/Lasso Regression (handle multicollinearity)
   • Decision Tree Regression (capture non-linearity)
   • Random Forest (ensemble learning)
   • Gradient Boosting (XGBoost, LightGBM)

3. Data Enhancement:
   • Collect more samples for better generalization
   • Include temporal features (year built, season)
   • Add external data (crime rates, school quality)

4. Model Optimization:
   • Cross-validation for robust evaluation
   • Feature selection techniques
   • Outlier detection and treatment
   • Regularization to prevent overfitting

KEY LEARNINGS:
--------------
✓ Linear regression provides a simple baseline for price prediction
✓ Data cleaning and preprocessing are crucial for model performance
✓ Train-test split prevents overfitting and validates generalization
✓ Multiple metrics (MSE, RMSE, R²) provide comprehensive evaluation
✓ Visualization helps understand model behavior and identify issues
✓ Understanding model limitations guides future improvements

CONCLUSION:
-----------
This project demonstrates fundamental machine learning workflow:
data preparation → model training → evaluation → interpretation.
While simple, linear regression provides valuable insights into feature
importance and serves as a foundation for more advanced techniques.
""".format(
    r2=r2,
    r2_percent=r2 * 100,
    interpretation="good predictive power" if r2 >= 0.7 else "moderate performance" if r2 >= 0.5 else "room for improvement",
    mse=mse,
    rmse=rmse
)

print(insights)

print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print()
print("Next Steps:")
print("  1. Review the visualizations in the 'visuals/' folder")
print("  2. Analyze the model coefficients to understand feature importance")
print("  3. Consider implementing suggested improvements")
print("  4. Document findings in the project report")
print()
print("Thank you for using this House Price Prediction model!")
print("=" * 70)
