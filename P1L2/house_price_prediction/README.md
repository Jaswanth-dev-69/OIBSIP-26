# 🏠 House Price Prediction using Linear Regression

## 📋 Project Overview

This project is part of the **Oasis Infobyte Data Analytics Internship (Level-2)**. The objective is to build a **Linear Regression model** to predict house prices using numerical features from a housing dataset.

The project demonstrates fundamental machine learning concepts including data preprocessing, feature selection, model training, evaluation, and result interpretation.

---

## 🎯 Project Objectives

- Understand and implement **Linear Regression** from scratch using Scikit-learn
- Perform comprehensive **data cleaning** and **preprocessing**
- Select relevant **numerical features** for prediction
- Train a Linear Regression model and evaluate its performance
- Visualize relationships between features and target variable
- Interpret model results with clear, actionable insights

---

## 📊 Dataset Description

**Source:** Kaggle  
**Reference:** [Housing Price Prediction – Linear Regression](https://www.kaggle.com/code/ashydv/housing-price-prediction-linear-regression)

### Dataset Characteristics:
- **Format:** CSV (Comma Separated Values)
- **Type:** Numerical features related to housing properties
- **Target Variable:** House price (continuous numerical value)
- **Features:** Various housing attributes (area, rooms, location factors, etc.)

### Key Features (Examples):
- House size/area
- Number of bedrooms/bathrooms
- Location-based factors
- Property age
- Other numerical attributes affecting house prices

> **Note:** The actual features depend on the specific dataset version. The script automatically identifies and uses all available numerical features.

---

## 🛠️ Tools & Libraries Used

| Library | Purpose |
|---------|---------|
| **Python 3.x** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization (plots and charts) |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn** | Machine learning model and evaluation |

### Scikit-learn Components:
- `train_test_split` - Splitting data into training and testing sets
- `LinearRegression` - Linear regression model implementation
- `mean_squared_error` - MSE evaluation metric
- `r2_score` - R-squared evaluation metric

---

## 📁 Project Structure

```
house_price_prediction/
│
├── data/
│   └── housing.csv                  # Dataset file
│
├── scripts/
│   └── house_price_prediction.py    # Main Python script
│
├── visuals/
│   ├── feature_vs_price.png         # Feature correlation visualization
│   └── actual_vs_predicted.png      # Model prediction visualization
│
└── README.md                        # Project documentation (this file)
```

---

## 🔍 Steps Followed

### 1️⃣ **Data Loading & Exploration**
- Loaded dataset using Pandas `read_csv()`
- Displayed first and last 5 rows using `head()` and `tail()`
- Checked dataset shape, column names, and data types
- Generated statistical summary using `describe()` and `info()`

### 2️⃣ **Data Cleaning**
- Identified and handled missing values
- Removed duplicate rows
- Ensured all feature columns are numeric
- Validated data integrity and consistency

**Cleaning Strategy:**
- Missing values: Dropped rows with null values (simple approach)
- Duplicates: Removed to avoid data leakage
- Data types: Ensured numerical features for regression

### 3️⃣ **Feature Selection**
- Identified target variable (house price)
- Separated independent variables (X) and dependent variable (y)
- Selected all relevant numerical features
- Analyzed correlation between features and target

**Justification:**
- All numerical features potentially influence house prices
- Linear regression automatically assigns appropriate weights
- Correlation analysis helps understand feature importance

### 4️⃣ **Train-Test Split**
- Split dataset: **80% training**, **20% testing**
- Used `random_state=42` for reproducibility
- Ensured unbiased model evaluation

**Why Train-Test Split?**
- **Training Set:** Teaches the model patterns in data
- **Testing Set:** Evaluates model on unseen data
- **Prevents Overfitting:** Ensures model generalizes well
- **Realistic Performance:** Provides true model accuracy

### 5️⃣ **Model Training**
- Created `LinearRegression` model instance
- Trained model using `fit()` on training data
- Generated model coefficients (β values) for each feature
- Calculated intercept (β₀)

### 6️⃣ **Model Evaluation**
- Made predictions on test data using `predict()`
- Calculated performance metrics:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared (R²) Score**

### 7️⃣ **Visualization**
Created and saved two key visualizations:

1. **Feature vs Price Scatter Plot**
   - Shows relationship between most correlated feature and price
   - Helps identify linear/non-linear patterns
   - Saved as: `visuals/feature_vs_price.png`

2. **Actual vs Predicted Prices**
   - Compares model predictions with actual prices
   - Includes perfect prediction reference line
   - Shows prediction accuracy visually
   - Saved as: `visuals/actual_vs_predicted.png`

---

## 📈 Evaluation Metrics Explained

### 1. **Mean Squared Error (MSE)**
- **Formula:** MSE = (1/n) × Σ(actual - predicted)²
- **Meaning:** Average squared difference between actual and predicted values
- **Range:** 0 to ∞ (lower is better)
- **Interpretation:** Penalizes larger errors more heavily

### 2. **Root Mean Squared Error (RMSE)**
- **Formula:** RMSE = √MSE
- **Meaning:** Average prediction error in same units as target variable
- **Range:** 0 to ∞ (lower is better)
- **Interpretation:** Easier to interpret than MSE (same scale as prices)

### 3. **R-squared (R²) Score**
- **Formula:** R² = 1 - (SS_res / SS_tot)
- **Meaning:** Proportion of variance explained by the model
- **Range:** -∞ to 1 (closer to 1 is better)
- **Interpretation:**
  - **R² = 1.0:** Perfect predictions
  - **R² = 0.9-1.0:** Excellent model
  - **R² = 0.7-0.9:** Good model
  - **R² = 0.5-0.7:** Moderate model
  - **R² < 0.5:** Poor model (needs improvement)

---

## 🎓 Key Learnings

### Technical Skills:
✅ **Data Preprocessing:** Handling missing values, duplicates, and data types  
✅ **Exploratory Data Analysis:** Understanding data through statistics and correlation  
✅ **Feature Engineering:** Selecting relevant features for modeling  
✅ **Model Training:** Implementing Linear Regression with Scikit-learn  
✅ **Model Evaluation:** Using multiple metrics for comprehensive assessment  
✅ **Data Visualization:** Creating meaningful plots to communicate insights  

### Machine Learning Concepts:
✅ **Supervised Learning:** Using labeled data to train predictive models  
✅ **Train-Test Split:** Importance of separating training and testing data  
✅ **Overfitting vs Underfitting:** Balancing model complexity and generalization  
✅ **Model Interpretation:** Understanding coefficients and their impact  
✅ **Performance Metrics:** Choosing appropriate evaluation criteria  

### Professional Skills:
✅ **Code Organization:** Writing clean, well-commented, maintainable code  
✅ **Documentation:** Creating comprehensive README for project clarity  
✅ **Best Practices:** Following industry-standard ML workflow  
✅ **Communication:** Presenting technical results to non-technical audiences  

---

## 🔬 Model Limitations

1. **Linearity Assumption**
   - Assumes linear relationship between features and price
   - Real-world housing prices often have non-linear patterns

2. **Outlier Sensitivity**
   - Extreme values can significantly skew predictions
   - May require outlier detection and treatment

3. **Multicollinearity**
   - Correlated features can affect coefficient interpretation
   - Doesn't necessarily reduce prediction accuracy

4. **Feature Interactions**
   - Cannot capture complex interactions between features
   - Example: Price impact of "large house in bad location"

5. **Constant Variance Assumption**
   - Assumes prediction error is consistent across all price ranges
   - May not hold for very cheap or very expensive houses

---

## 🚀 Future Improvements

### Data Enhancements:
- Add categorical features (neighborhood, house style, etc.)
- Include temporal features (season, market trends)
- Incorporate external data (crime rates, school quality)
- Collect more samples for better generalization

### Feature Engineering:
- Create polynomial features for non-linear relationships
- Generate interaction features (e.g., area × location)
- Apply feature scaling/normalization
- Use feature selection techniques (RFE, Lasso)

### Advanced Modeling:
- **Ridge Regression:** Handle multicollinearity
- **Lasso Regression:** Automatic feature selection
- **Polynomial Regression:** Capture non-linear patterns
- **Decision Trees:** Handle complex interactions
- **Random Forest:** Ensemble learning for better accuracy
- **Gradient Boosting:** XGBoost, LightGBM for state-of-the-art performance

### Optimization Techniques:
- Cross-validation (k-fold) for robust evaluation
- Hyperparameter tuning
- Outlier detection and treatment
- Regularization to prevent overfitting

---

## 💡 Insights & Interpretation

### Model Performance:
The Linear Regression model provides a **baseline prediction** for house prices. The R² score indicates how well the selected features explain price variance.

### Feature Importance:
Model coefficients reveal which features most strongly influence prices:
- **Positive coefficients:** Feature increases price
- **Negative coefficients:** Feature decreases price
- **Magnitude:** Indicates strength of relationship

### Practical Applications:
- **Real Estate Valuation:** Quick price estimates for properties
- **Market Analysis:** Understanding price-driving factors
- **Investment Decisions:** Identifying undervalued/overvalued properties
- **Price Negotiation:** Data-driven pricing strategies

---

## 📝 How to Run This Project

### Prerequisites:
```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Steps:
1. **Download the dataset** from Kaggle
2. **Place** `housing.csv` in the `data/` folder
3. **Navigate** to the `scripts/` folder
4. **Run** the Python script:
   ```bash
   python house_price_prediction.py
   ```
5. **View** generated visualizations in the `visuals/` folder

---

## 👨‍💻 Author

**[Your Name]**  
Data Analytics Intern - Oasis Infobyte (Level 2)  
Project: House Price Prediction using Linear Regression  
Date: January 14, 2026

---

## 📧 Contact

For questions or feedback regarding this project:
- **Email:** [your.email@example.com]
- **LinkedIn:** [Your LinkedIn Profile]
- **GitHub:** [Your GitHub Profile]

---

## 📄 License

This project is created for educational purposes as part of the Oasis Infobyte internship program.

---

## 🙏 Acknowledgments

- **Oasis Infobyte** for the internship opportunity
- **Kaggle** for providing the housing dataset
- **Scikit-learn** community for excellent documentation and tools
- **Python** data science community for open-source libraries

---

## 📚 References

1. [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
2. [Kaggle Housing Price Dataset](https://www.kaggle.com/code/ashydv/housing-price-prediction-linear-regression)
3. [Understanding R-squared](https://blog.minitab.com/en/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)
4. [Linear Regression Assumptions](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/)

---

**⭐ If you found this project helpful, please star the repository!**

---

*Last Updated: January 14, 2026*
