# 🎬 Project 3: Data Cleaning - YouTube Trending Videos Dataset

## 📋 Project Overview

This project is part of the **Oasis Infobyte Data Analytics Internship** program. The objective is to perform **end-to-end data cleaning** on a real-world dataset to ensure data accuracy, consistency, reliability, and analysis-ready quality.

**Focus Area:** Data Cleaning (No machine learning or predictive modeling)

---

## 📊 Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle |
| **Dataset Name** | YouTube Trending Videos Dataset |
| **Dataset Link** | [https://www.kaggle.com/datasets/datasnaek/youtube-new](https://www.kaggle.com/datasets/datasnaek/youtube-new) |
| **File Used** | USvideos.csv |
| **Loading Method** | KaggleHub with KaggleDatasetAdapter.PANDAS |

### Dataset Contents
The dataset contains information about trending YouTube videos including:
- 📹 **Video Metadata**: video_id, title, channel_title, description, tags
- 📈 **Engagement Metrics**: views, likes, dislikes, comment_count
- 📂 **Categorization**: category_id
- 📅 **Temporal Data**: trending_date, publish_time
- ⚙️ **Status Flags**: comments_disabled, ratings_disabled, video_error_or_removed

---

## 🔧 Tools & Libraries Used

| Tool/Library | Purpose |
|--------------|---------|
| **Python 3.x** | Primary programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical operations |
| **Matplotlib** | Outlier visualization |
| **Seaborn** | Enhanced visualizations |
| **KaggleHub** | Dataset loading from Kaggle |

---

## 📁 Project Structure

```
data_cleaning_youtube/
├── 📂 data/
│   ├── 📂 raw/
│   │   └── 📄 youtube_raw.csv          # Original dataset
│   └── 📂 cleaned/
│       └── 📄 youtube_cleaned.csv      # Cleaned dataset
├── 📂 scripts/
│   └── 📄 data_cleaning.py             # Main cleaning script
├── 📂 visuals/
│   └── 📄 outliers.png                 # Outlier visualization
└── 📄 README.md                        # Project documentation
```

---

## 🛠️ Data Cleaning Steps Performed

### Task 1: Data Loading ✅
- Loaded dataset from Kaggle using `kagglehub`
- Displayed initial exploration: `head()`, `tail()`, `shape`, `columns`, `info()`
- Saved raw data for reference

### Task 2: Data Integrity Checks ✅
- Verified data types for all columns
- Identified inconsistent date formats (YY.DD.MM format)
- Checked for impossible values:
  - Negative views, likes, dislikes, comments
  - Likes exceeding views
  - Dislikes exceeding views

### Task 3: Missing Data Handling ✅
- Identified null values using `isnull()` and `isnull().sum()`
- Applied column-specific strategies:
  | Column Type | Strategy | Reason |
  |-------------|----------|--------|
  | Numerical | Fill with **median** | Robust to outliers/skewed data |
  | Categorical | Fill with **mode** | Most frequent value |
  | Text (title, description) | Fill with **'Not Available'** | Preserve row, indicate missing |

### Task 4: Duplicate Removal ✅
- Identified duplicate rows
- Removed duplicates (kept first occurrence)
- Documented before vs after row counts

### Task 5: Data Standardization ✅
- **Text Columns**:
  - Stripped leading/trailing whitespace
  - Converted tags to lowercase
  - Preserved channel name casing (brand identity)
- **Date Columns**:
  - Converted `trending_date` from YY.DD.MM to datetime64
  - Converted `publish_time` to datetime64
- **Numeric Columns**:
  - Ensured views, likes, dislikes, comment_count are integers
- **Boolean Columns**:
  - Converted flag columns to proper boolean type

### Task 6: Outlier Detection ✅
- **Method Used**: IQR (Interquartile Range)
  - Lower Bound = Q1 - 1.5 × IQR
  - Upper Bound = Q3 + 1.5 × IQR
- **Visualization**: Box plots for views, likes, dislikes, comment_count
- **Decision**: **Retained outliers**
  - Reason: Viral content is natural in YouTube data
  - High engagement metrics represent genuinely popular content
  - Removing them would bias analysis toward average-performing videos

### Task 7: Final Validation ✅
- Rechecked for null values (none remaining)
- Rechecked for duplicates (none remaining)
- Verified all data types are correct
- Saved cleaned dataset

---

## 📈 Before vs After Improvements

| Metric | Before Cleaning | After Cleaning | Improvement |
|--------|-----------------|----------------|-------------|
| **Null Values** | Present in description/other fields | 0 | ✅ 100% resolved |
| **Duplicates** | Multiple duplicate rows | 0 | ✅ 100% removed |
| **Date Format** | Inconsistent (YY.DD.MM) | Standardized datetime64 | ✅ Proper formatting |
| **Data Types** | Mixed/incorrect types | Appropriate types per column | ✅ Type consistency |
| **Text Data** | Inconsistent whitespace/casing | Cleaned and standardized | ✅ Consistency |
| **Numeric Data** | Potential type issues | All integers validated | ✅ Type safety |

---

## 🎯 Why Data Cleaning is Important in Analytics

Data cleaning is a **critical foundation** for any data analytics project. Here's why:

### 1. 📊 Ensures Accurate Analysis
- Dirty data leads to incorrect insights and flawed conclusions
- Clean data produces reliable, trustworthy results

### 2. 🔄 Improves Consistency
- Standardized formats (dates, text, numbers) enable proper comparisons
- Eliminates confusion from inconsistent representations

### 3. ⚡ Enhances Efficiency
- Analysts spend less time debugging data issues
- Automated processes run without unexpected errors

### 4. 📈 Supports Better Decision Making
- Business decisions based on clean data are more reliable
- Reduces risk of costly mistakes from data errors

### 5. 🤝 Enables Data Integration
- Clean, standardized data can be easily merged with other sources
- Facilitates cross-platform analytics

### 6. 🔒 Maintains Data Integrity
- Removes duplicates that could skew statistics
- Handles missing values appropriately

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn kagglehub
```

### Configure Kaggle API
Ensure your Kaggle API token is configured:
1. Download `kaggle.json` from Kaggle account settings
2. Place in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### Run the Script
```bash
cd data_cleaning_youtube/scripts
python data_cleaning.py
```

### Output Files
- Raw data saved to: `data/raw/youtube_raw.csv`
- Cleaned data saved to: `data/cleaned/youtube_cleaned.csv`
- Outlier visualization saved to: `visuals/outliers.png`

---

## 📝 Key Learnings

1. **Always explore data first** before applying any cleaning operations
2. **Document every decision** with clear reasoning
3. **Domain knowledge matters** - understanding YouTube metrics helped in outlier decisions
4. **Use appropriate strategies** for different data types (median for skewed numerical data)
5. **Validate after each step** to ensure cleaning worked correctly

---

## 👤 Author

**Data Analytics Intern**  
Oasis Infobyte Internship Program  
Project 3: Data Cleaning  
January 2026

---

## 📜 License

This project is created for educational purposes as part of the Oasis Infobyte Data Analytics Internship.

---

## 🙏 Acknowledgments

- **Oasis Infobyte** for the internship opportunity
- **Kaggle** for providing the YouTube Trending Videos dataset
- **Mitchell J** (Dataset Creator) for compiling the original data
