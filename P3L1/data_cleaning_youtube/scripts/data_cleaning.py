"""
================================================================================
PROJECT 3: DATA CLEANING - YOUTUBE TRENDING VIDEOS DATASET
================================================================================
Oasis Infobyte Data Analytics Internship
Author: Data Analytics Intern
Date: January 2026

Objective: Perform end-to-end data cleaning to ensure data accuracy, 
consistency, reliability, and analysis-ready quality.
================================================================================
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# =============================================================================
# DEFINE PROJECT PATHS
# =============================================================================
# Get the script's directory and set up relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'raw', 'youtube_raw.csv')
CLEANED_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'cleaned', 'youtube_cleaned.csv')
VISUALS_PATH = os.path.join(PROJECT_DIR, 'visuals', 'outliers.png')

print("=" * 80)
print("PROJECT 3: DATA CLEANING - YOUTUBE TRENDING VIDEOS DATASET")
print("=" * 80)

# =============================================================================
# TASK 1: DATA LOADING
# =============================================================================
print("\n" + "=" * 80)
print("TASK 1: DATA LOADING")
print("=" * 80)

# Load dataset from Kaggle using kagglehub
# The dataset contains YouTube trending video statistics from multiple countries
print("\n📥 Loading dataset from Kaggle...")

try:
    # First, download the dataset using kagglehub
    dataset_path = kagglehub.dataset_download("datasnaek/youtube-new")
    print(f"✅ Dataset downloaded to: {dataset_path}")
    
    # Load the CSV with proper encoding (the file contains special characters)
    csv_file = os.path.join(dataset_path, "USvideos.csv")
    df = pd.read_csv(csv_file, encoding='latin-1', on_bad_lines='skip')
    print("✅ Dataset loaded successfully from Kaggle!")
except Exception as e:
    print(f"⚠️ Error loading from Kaggle: {e}")
    print("Attempting to load from local raw data path...")
    df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1', on_bad_lines='skip')

# Save raw data for reference
print(f"\n💾 Saving raw data to: {RAW_DATA_PATH}")
df.to_csv(RAW_DATA_PATH, index=False)

# Display basic information about the dataset
print("\n" + "-" * 40)
print("📊 DATASET OVERVIEW")
print("-" * 40)

# Display first 5 rows
print("\n🔹 First 5 rows (head):")
print(df.head())

# Display last 5 rows
print("\n🔹 Last 5 rows (tail):")
print(df.tail())

# Display shape
print(f"\n🔹 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Display column names
print("\n🔹 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

# Display data types and memory usage
print("\n🔹 Data Types & Info:")
print(df.info())

# Store initial statistics for comparison
initial_rows = df.shape[0]
initial_cols = df.shape[1]
initial_nulls = df.isnull().sum().sum()
initial_duplicates = df.duplicated().sum()

print(f"\n📈 Initial Statistics:")
print(f"   • Total Rows: {initial_rows:,}")
print(f"   • Total Columns: {initial_cols}")
print(f"   • Total Null Values: {initial_nulls:,}")
print(f"   • Duplicate Rows: {initial_duplicates:,}")

# =============================================================================
# TASK 2: DATA INTEGRITY CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("TASK 2: DATA INTEGRITY CHECKS")
print("=" * 80)

# Check 2.1: Verify Data Types
print("\n🔹 Current Data Types:")
print(df.dtypes)

# Check 2.2: Identify columns that need type conversion
print("\n🔹 Analyzing data type issues...")

# Check date columns - they should be datetime, not object
date_columns = ['trending_date', 'publish_time']
for col in date_columns:
    if col in df.columns:
        print(f"   • '{col}' is currently: {df[col].dtype} → Should be: datetime64")

# Check numeric columns
numeric_columns = ['views', 'likes', 'dislikes', 'comment_count']
for col in numeric_columns:
    if col in df.columns:
        print(f"   • '{col}' is currently: {df[col].dtype}")

# Check 2.3: Identify impossible/invalid values
print("\n🔹 Checking for impossible values...")

issues_found = []

# Check for negative values in numeric columns (should not exist)
for col in numeric_columns:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            issues_found.append(f"   ⚠️ '{col}' has {negative_count} negative values")
            print(f"   ⚠️ '{col}' has {negative_count} negative values")
        else:
            print(f"   ✅ '{col}' has no negative values")

# Check if likes > views (logically impossible in most cases)
if 'likes' in df.columns and 'views' in df.columns:
    likes_exceed_views = (df['likes'] > df['views']).sum()
    if likes_exceed_views > 0:
        issues_found.append(f"   ⚠️ {likes_exceed_views} rows where likes > views")
        print(f"   ⚠️ {likes_exceed_views} rows where likes > views (data inconsistency)")
    else:
        print(f"   ✅ All likes are ≤ views")

# Check if dislikes > views
if 'dislikes' in df.columns and 'views' in df.columns:
    dislikes_exceed_views = (df['dislikes'] > df['views']).sum()
    if dislikes_exceed_views > 0:
        issues_found.append(f"   ⚠️ {dislikes_exceed_views} rows where dislikes > views")
        print(f"   ⚠️ {dislikes_exceed_views} rows where dislikes > views")
    else:
        print(f"   ✅ All dislikes are ≤ views")

# Check for inconsistent date formats
print("\n🔹 Checking date format consistency...")
if 'trending_date' in df.columns:
    # Sample some dates to check format
    sample_dates = df['trending_date'].head(5).tolist()
    print(f"   Sample trending_date values: {sample_dates}")
    # The format appears to be YY.DD.MM which is unusual and needs standardization

if 'publish_time' in df.columns:
    sample_publish = df['publish_time'].head(5).tolist()
    print(f"   Sample publish_time values: {sample_publish}")

if not issues_found:
    print("\n✅ No major data integrity issues found!")
else:
    print(f"\n⚠️ Found {len(issues_found)} integrity issues to address")

# =============================================================================
# TASK 3: MISSING DATA HANDLING
# =============================================================================
print("\n" + "=" * 80)
print("TASK 3: MISSING DATA HANDLING")
print("=" * 80)

# Check for missing values
print("\n🔹 Missing Values Summary:")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing %': missing_percent.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("   No missing values found!")

# Handle missing values based on column type
print("\n🔹 Handling Missing Values...")

# Strategy for each column type:
# - Numerical columns: Fill with median (robust to outliers)
# - Categorical columns: Fill with mode or 'Unknown'
# - Description/Text: Fill with 'No description available'

for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        if df[col].dtype in ['int64', 'float64']:
            # NUMERICAL: Fill with median (more robust than mean for skewed data)
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   ✅ '{col}': Filled {null_count} nulls with median ({median_val:.2f})")
            # Reason: Median is chosen because view/like counts are often right-skewed
            
        elif df[col].dtype == 'object':
            if col in ['description', 'title']:
                # TEXT FIELDS: Fill with placeholder
                df[col].fillna('Not Available', inplace=True)
                print(f"   ✅ '{col}': Filled {null_count} nulls with 'Not Available'")
                # Reason: Preserves row while indicating missing content
                
            else:
                # CATEGORICAL: Fill with mode (most frequent value)
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"   ✅ '{col}': Filled {null_count} nulls with mode ('{mode_val}')")
                # Reason: Mode represents the most common category

# Verify no missing values remain
remaining_nulls = df.isnull().sum().sum()
print(f"\n📊 Missing Values After Cleaning: {remaining_nulls}")

# =============================================================================
# TASK 4: DUPLICATE REMOVAL
# =============================================================================
print("\n" + "=" * 80)
print("TASK 4: DUPLICATE REMOVAL")
print("=" * 80)

# Count duplicates before removal
duplicates_before = df.duplicated().sum()
rows_before = len(df)

print(f"\n🔹 Duplicate Analysis:")
print(f"   • Total rows before: {rows_before:,}")
print(f"   • Duplicate rows found: {duplicates_before:,}")
print(f"   • Duplicate percentage: {(duplicates_before/rows_before)*100:.2f}%")

if duplicates_before > 0:
    # Show sample of duplicate rows
    print("\n🔹 Sample duplicate rows:")
    duplicate_rows = df[df.duplicated(keep=False)].head(4)
    print(duplicate_rows[['video_id', 'title', 'views']].head(4) if 'video_id' in df.columns else duplicate_rows.head(4))
    
    # Remove duplicates - keep first occurrence
    # Reason: First occurrence is typically the earliest trending instance
    df.drop_duplicates(inplace=True, keep='first')
    
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    print(f"\n✅ Duplicate Removal Complete:")
    print(f"   • Rows before: {rows_before:,}")
    print(f"   • Rows after: {rows_after:,}")
    print(f"   • Rows removed: {rows_removed:,}")
else:
    print("\n✅ No duplicate rows found!")

# =============================================================================
# TASK 5: STANDARDIZATION
# =============================================================================
print("\n" + "=" * 80)
print("TASK 5: DATA STANDARDIZATION")
print("=" * 80)

# 5.1: Standardize Text Columns
print("\n🔹 Standardizing Text Columns...")

# Convert text columns to consistent case and clean whitespace
text_columns = ['title', 'channel_title', 'tags', 'description']
for col in text_columns:
    if col in df.columns:
        # Strip leading/trailing whitespace
        df[col] = df[col].astype(str).str.strip()
        
        # For channel_title, use title case for consistency
        if col == 'channel_title':
            # Keep original case for channel names (brand identity)
            df[col] = df[col].str.strip()
            print(f"   ✅ '{col}': Stripped whitespace, preserved original case")
        elif col == 'tags':
            # Tags: convert to lowercase for consistency
            df[col] = df[col].str.lower()
            print(f"   ✅ '{col}': Converted to lowercase, stripped whitespace")
        else:
            # Title and description: strip whitespace only
            print(f"   ✅ '{col}': Stripped whitespace")

# 5.2: Standardize Category Names
print("\n🔹 Standardizing Category IDs...")
if 'category_id' in df.columns:
    # Category ID should be integer
    df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(0).astype(int)
    unique_categories = df['category_id'].nunique()
    print(f"   ✅ 'category_id': Converted to integer, {unique_categories} unique categories")

# 5.3: Convert Date/Time Columns to Proper DateTime Format
print("\n🔹 Standardizing Date/Time Columns...")

# Convert trending_date (format: YY.DD.MM → proper datetime)
if 'trending_date' in df.columns:
    try:
        # The trending_date appears to be in format: YY.DD.MM
        df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
        print(f"   ✅ 'trending_date': Converted to datetime64")
        print(f"      Date range: {df['trending_date'].min()} to {df['trending_date'].max()}")
    except Exception as e:
        print(f"   ⚠️ 'trending_date' conversion issue: {e}")

# Convert publish_time to datetime
if 'publish_time' in df.columns:
    try:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        print(f"   ✅ 'publish_time': Converted to datetime64")
        print(f"      Date range: {df['publish_time'].min()} to {df['publish_time'].max()}")
    except Exception as e:
        print(f"   ⚠️ 'publish_time' conversion issue: {e}")

# 5.4: Ensure Numeric Columns are Numeric
print("\n🔹 Ensuring Numeric Columns are Properly Typed...")

numeric_columns = ['views', 'likes', 'dislikes', 'comment_count']
for col in numeric_columns:
    if col in df.columns:
        # Convert to numeric, coercing errors to NaN, then fill with 0
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        print(f"   ✅ '{col}': {original_dtype} → int64")

# 5.5: Clean Boolean Columns
print("\n🔹 Standardizing Boolean Columns...")
bool_columns = ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']
for col in bool_columns:
    if col in df.columns:
        # Ensure boolean columns are properly typed
        df[col] = df[col].astype(bool)
        print(f"   ✅ '{col}': Converted to boolean")

print("\n📊 Updated Data Types:")
print(df.dtypes)

# =============================================================================
# TASK 6: OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("TASK 6: OUTLIER DETECTION")
print("=" * 80)

# Define numeric columns for outlier analysis
outlier_columns = ['views', 'likes', 'dislikes', 'comment_count']
outlier_columns = [col for col in outlier_columns if col in df.columns]

print("\n🔹 Detecting Outliers using IQR Method...")
print("   (IQR = Q3 - Q1; Outliers are values < Q1-1.5*IQR or > Q3+1.5*IQR)")

outlier_summary = {}

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers_low = (df[col] < lower_bound).sum()
    outliers_high = (df[col] > upper_bound).sum()
    total_outliers = outliers_low + outliers_high
    outlier_percent = (total_outliers / len(df)) * 100
    
    outlier_summary[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Low Outliers': outliers_low,
        'High Outliers': outliers_high,
        'Total Outliers': total_outliers,
        'Outlier %': outlier_percent
    }
    
    print(f"\n   📊 {col}:")
    print(f"      Q1: {Q1:,.0f} | Q3: {Q3:,.0f} | IQR: {IQR:,.0f}")
    print(f"      Lower Bound: {lower_bound:,.0f} | Upper Bound: {upper_bound:,.0f}")
    print(f"      Low Outliers: {outliers_low:,} | High Outliers: {outliers_high:,}")
    print(f"      Total Outliers: {total_outliers:,} ({outlier_percent:.2f}%)")

# 6.1: Visualize Outliers
print("\n🔹 Creating Outlier Visualization...")

# Set up the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Outlier Detection - YouTube Trending Videos\n(Box Plots showing distribution and outliers)', 
             fontsize=14, fontweight='bold')

# Color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for idx, col in enumerate(outlier_columns):
    ax = axes[idx // 2, idx % 2]
    
    # Create boxplot
    bp = ax.boxplot(df[col].dropna(), patch_artist=True, 
                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
    
    ax.set_title(f'{col.replace("_", " ").title()}\n(Outliers: {outlier_summary[col]["Total Outliers"]:,})', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = f'Median: {df[col].median():,.0f}\nMean: {df[col].mean():,.0f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(VISUALS_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"   ✅ Outlier visualization saved to: {VISUALS_PATH}")

# 6.2: Outlier Treatment Decision
print("\n🔹 Outlier Treatment Decision:")
print("""
   DECISION: RETAIN OUTLIERS (with justification)
   
   Reasoning:
   ─────────────────────────────────────────────────────────────────
   1. VIRAL CONTENT IS NATURAL: In YouTube data, extremely high views,
      likes, and comments are EXPECTED for viral videos. These are not
      errors but represent genuinely popular content.
   
   2. DOMAIN RELEVANCE: For trending video analysis, removing viral 
      videos would eliminate the most important data points that define
      what "trending" means.
   
   3. DATA INTEGRITY: These values are genuine metrics from YouTube's
      platform, not measurement errors or data entry mistakes.
   
   4. ANALYSIS IMPACT: Removing outliers would skew results toward
      average-performing videos, losing insights about viral success.
   
   ALTERNATIVE ACTIONS (if needed for specific analyses):
   • Use log transformation for visualizations
   • Use median instead of mean for central tendency
   • Apply winsorization (capping) only for modeling purposes
   ─────────────────────────────────────────────────────────────────
""")

# Optional: Show what capping would look like (without applying)
print("   📊 If capping were applied (for reference only):")
for col in outlier_columns:
    upper = outlier_summary[col]['Upper Bound']
    affected = outlier_summary[col]['High Outliers']
    max_val = df[col].max()
    print(f"      • {col}: Would cap {affected:,} values from max {max_val:,} to {upper:,.0f}")

# =============================================================================
# TASK 7: FINAL VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("TASK 7: FINAL VALIDATION")
print("=" * 80)

# 7.1: Recheck Null Values
print("\n🔹 Final Null Value Check:")
final_nulls = df.isnull().sum()
if final_nulls.sum() == 0:
    print("   ✅ No null values remaining")
else:
    print(final_nulls[final_nulls > 0])

# 7.2: Recheck Duplicates
print("\n🔹 Final Duplicate Check:")
final_duplicates = df.duplicated().sum()
if final_duplicates == 0:
    print("   ✅ No duplicate rows remaining")
else:
    print(f"   ⚠️ {final_duplicates} duplicate rows found")

# 7.3: Verify Data Types
print("\n🔹 Final Data Types:")
print(df.dtypes)

# 7.4: Final Dataset Statistics
print("\n🔹 Final Dataset Statistics:")
print(df.describe())

# 7.5: Save Cleaned Dataset
print("\n" + "-" * 40)
print("💾 SAVING CLEANED DATASET")
print("-" * 40)

df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"   ✅ Cleaned dataset saved to: {CLEANED_DATA_PATH}")
print(f"   📊 Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# CLEANING SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("📈 DATA CLEANING SUMMARY")
print("=" * 80)

final_rows = df.shape[0]
final_cols = df.shape[1]
final_nulls_total = df.isnull().sum().sum()
final_duplicates_total = df.duplicated().sum()

print(f"""
   ┌─────────────────────────────────────────────────────────────┐
   │                    BEFORE vs AFTER                          │
   ├─────────────────────────────────────────────────────────────┤
   │  Metric              │   Before    │   After    │ Change    │
   ├─────────────────────────────────────────────────────────────┤
   │  Total Rows          │  {initial_rows:>8,}   │  {final_rows:>8,}  │ {final_rows - initial_rows:>+8,}  │
   │  Total Columns       │  {initial_cols:>8}   │  {final_cols:>8}  │ {final_cols - initial_cols:>+8}  │
   │  Null Values         │  {initial_nulls:>8,}   │  {final_nulls_total:>8,}  │ {final_nulls_total - initial_nulls:>+8,}  │
   │  Duplicate Rows      │  {initial_duplicates:>8,}   │  {final_duplicates_total:>8,}  │ {final_duplicates_total - initial_duplicates:>+8,}  │
   └─────────────────────────────────────────────────────────────┘

   ✅ CLEANING TASKS COMPLETED:
   ─────────────────────────────────────────────────────────────
   [✓] Task 1: Data loaded and explored
   [✓] Task 2: Data integrity verified
   [✓] Task 3: Missing values handled
   [✓] Task 4: Duplicates removed
   [✓] Task 5: Data standardized (dates, text, types)
   [✓] Task 6: Outliers detected and analyzed
   [✓] Task 7: Final validation completed
   
   📁 OUTPUT FILES:
   ─────────────────────────────────────────────────────────────
   • Raw Data:     {RAW_DATA_PATH}
   • Cleaned Data: {CLEANED_DATA_PATH}
   • Visualizations: {VISUALS_PATH}
""")

print("=" * 80)
print("🎉 DATA CLEANING COMPLETE - DATASET IS NOW ANALYSIS-READY!")
print("=" * 80)
