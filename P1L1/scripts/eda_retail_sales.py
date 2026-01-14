"""
Exploratory Data Analysis (EDA) on Retail Sales Data
Oasis Infobyte - Data Analytics Internship

Author: Data Analytics Intern
Date: January 2026
"""

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# Setup folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VISUALS_DIR = os.path.join(PROJECT_DIR, 'visuals')
RESULTS_FILE = os.path.join(PROJECT_DIR, 'eda_results.txt')

# Create visuals folder if needed
if not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)

# Set plot style
sns.set_style("whitegrid")

# Open results file
results = open(RESULTS_FILE, 'w', encoding='utf-8')

def log(text):
    """Print to screen and save to file"""
    print(text)
    results.write(text + '\n')

# ============================================
# STEP 2: LOAD DATASET
# ============================================

log("=" * 50)
log("RETAIL SALES DATA - EDA REPORT")
log("=" * 50)

log("\n>> Loading dataset from Kaggle...")

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mohammadtalib786/retail-sales-dataset",
    "retail_sales_dataset.csv",
)

log("Dataset loaded!")
log(f"Rows: {df.shape[0]}")
log(f"Columns: {df.shape[1]}")

# ============================================
# STEP 3: VIEW THE DATA
# ============================================

log("\n" + "=" * 50)
log("STEP 3: VIEW THE DATA")
log("=" * 50)

log("\n--- First 5 Rows ---")
log(df.head().to_string())

log("\n--- Column Names ---")
log(str(df.columns.tolist()))

log("\n--- Data Types ---")
log(df.dtypes.to_string())

# ============================================
# STEP 4: CHECK MISSING VALUES
# ============================================

log("\n" + "=" * 50)
log("STEP 4: CHECK MISSING VALUES")
log("=" * 50)

log(df.isnull().sum().to_string())
log(f"\nTotal Missing: {df.isnull().sum().sum()}")

# ============================================
# STEP 5: CHECK DUPLICATES
# ============================================

log("\n" + "=" * 50)
log("STEP 5: CHECK DUPLICATES")
log("=" * 50)

duplicates = df.duplicated().sum()
log(f"Duplicate Rows: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    log(f"Removed {duplicates} duplicates")

# ============================================
# STEP 6: UNIQUE VALUES
# ============================================

log("\n" + "=" * 50)
log("STEP 6: UNIQUE VALUES")
log("=" * 50)

log(df.nunique().to_string())

# ============================================
# STEP 7: CONVERT DATE
# ============================================

log("\n" + "=" * 50)
log("STEP 7: CONVERT DATE")
log("=" * 50)

df['Date'] = pd.to_datetime(df['Date'])
log("Date column converted!")

# ============================================
# STEP 8: DESCRIPTIVE STATISTICS
# ============================================

log("\n" + "=" * 50)
log("STEP 8: DESCRIPTIVE STATISTICS")
log("=" * 50)

log("\n--- describe() output ---")
log(df.describe().to_string())

# Age stats
log("\n--- Age ---")
log(f"Mean: {df['Age'].mean():.2f}")
log(f"Median: {df['Age'].median():.2f}")
log(f"Mode: {df['Age'].mode()[0]}")
log(f"Std: {df['Age'].std():.2f}")

# Sales stats
log("\n--- Total Amount ---")
log(f"Mean: ${df['Total Amount'].mean():.2f}")
log(f"Median: ${df['Total Amount'].median():.2f}")
log(f"Mode: ${df['Total Amount'].mode()[0]}")
log(f"Std: ${df['Total Amount'].std():.2f}")

# ============================================
# STEP 9: TIME SERIES ANALYSIS
# ============================================

log("\n" + "=" * 50)
log("STEP 9: TIME SERIES ANALYSIS")
log("=" * 50)

df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby('Month')['Total Amount'].sum()

log("\n--- Monthly Sales ---")
log(monthly_sales.to_string())

# Line Plot
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='blue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '01_monthly_sales_trend.png'))
plt.close()
log("\nSaved: 01_monthly_sales_trend.png")

# ============================================
# STEP 10: CUSTOMER ANALYSIS
# ============================================

log("\n" + "=" * 50)
log("STEP 10: CUSTOMER ANALYSIS")
log("=" * 50)

# Gender
log("\n--- Gender Count ---")
log(df['Gender'].value_counts().to_string())

gender_sales = df.groupby('Gender')['Total Amount'].sum()
log("\n--- Sales by Gender ---")
log(gender_sales.to_string())

# Bar Plot - Gender
plt.figure(figsize=(8, 5))
sns.barplot(x=gender_sales.index, y=gender_sales.values)
plt.title('Sales by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '02_sales_by_gender.png'))
plt.close()
log("\nSaved: 02_sales_by_gender.png")

# Histogram - Age
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=15, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '03_age_distribution.png'))
plt.close()
log("\nSaved: 03_age_distribution.png")

# ============================================
# STEP 11: PRODUCT ANALYSIS
# ============================================

log("\n" + "=" * 50)
log("STEP 11: PRODUCT ANALYSIS")
log("=" * 50)

log("\n--- Category Count ---")
log(df['Product Category'].value_counts().to_string())

category_sales = df.groupby('Product Category')['Total Amount'].sum()
log("\n--- Sales by Category ---")
log(category_sales.to_string())

# Bar Plot - Category
plt.figure(figsize=(10, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Sales by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '04_sales_by_category.png'))
plt.close()
log("\nSaved: 04_sales_by_category.png")

# ============================================
# STEP 12: SALES DISTRIBUTION
# ============================================

log("\n" + "=" * 50)
log("STEP 12: SALES DISTRIBUTION")
log("=" * 50)

log(f"Total Sales: ${df['Total Amount'].sum():,.2f}")
log(f"Average Sale: ${df['Total Amount'].mean():,.2f}")
log(f"Min Sale: ${df['Total Amount'].min():,.2f}")
log(f"Max Sale: ${df['Total Amount'].max():,.2f}")

# Histogram - Sales
plt.figure(figsize=(10, 5))
sns.histplot(df['Total Amount'], bins=20, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Total Amount')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '05_sales_distribution.png'))
plt.close()
log("\nSaved: 05_sales_distribution.png")

# ============================================
# STEP 13: CORRELATION HEATMAP
# ============================================

log("\n" + "=" * 50)
log("STEP 13: CORRELATION ANALYSIS")
log("=" * 50)

numeric_cols = ['Age', 'Quantity', 'Price per Unit', 'Total Amount']
correlation = df[numeric_cols].corr()

log("\n--- Correlation Matrix ---")
log(correlation.to_string())

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, '06_correlation_heatmap.png'))
plt.close()
log("\nSaved: 06_correlation_heatmap.png")

# ============================================
# STEP 14: FINAL SUMMARY
# ============================================

log("\n" + "=" * 50)
log("FINAL SUMMARY")
log("=" * 50)

log(f"\nTotal Records: {df.shape[0]}")
log(f"Total Customers: {df['Customer ID'].nunique()}")
log(f"Product Categories: {df['Product Category'].nunique()}")
log(f"Total Revenue: ${df['Total Amount'].sum():,.2f}")
log(f"Average Sale: ${df['Total Amount'].mean():,.2f}")

# ============================================
# BUSINESS RECOMMENDATIONS
# ============================================

log("\n" + "=" * 50)
log("BUSINESS RECOMMENDATIONS")
log("=" * 50)

recommendations = """
1. FOCUS ON ELECTRONICS:
   Electronics has highest sales - stock more items.

2. TARGET FEMALE CUSTOMERS:
   Females buy slightly more - run female-focused ads.

3. PEAK MONTHS:
   May and October have high sales - prepare inventory.

4. SLOW MONTHS:
   September is slow - offer discounts to boost sales.

5. BUNDLE PRODUCTS:
   Price and Total Amount are linked - create bundles.

6. LOYALTY PROGRAM:
   Each customer bought once - start a loyalty program.
"""

log(recommendations)

log("\n" + "=" * 50)
log("EDA COMPLETED!")
log("=" * 50)

# Close the file
results.close()

print("\n" + "=" * 50)
print("ALL DONE!")
print("=" * 50)
print(f"\nResults saved to: eda_results.txt")
print(f"Charts saved to: visuals/ folder")
print("=" * 50)
