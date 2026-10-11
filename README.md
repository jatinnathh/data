# Electric Vehicle Purchase Prediction

> **Kaggle Binary Classification Competition** — Predict whether a consumer will purchase an Electric Vehicle based on demographic, behavioral, and infrastructure features.

| Metric | Score |
|--------|-------|
| **Public Leaderboard (ROC-AUC)** | **0.94132** |
| **Private Leaderboard (ROC-AUC)** | **0.94070** |

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Structure](#project-structure)
3. [Dataset Overview](#dataset-overview)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Best Model — Stacking Ensemble](#best-model--stacking-ensemble)
8. [Inference Pipeline](#inference-pipeline)
9. [Quick Start](#quick-start)
10. [Tech Stack](#tech-stack)
11. [Author](#author)

---

## Problem Statement

Given a set of consumer attributes — including annual income, daily commute distance, access to charging infrastructure, environmental concern level, and current vehicle ownership — the goal is to predict the **probability** that a given customer will purchase an Electric Vehicle.

- **Task:** Binary Classification
- **Evaluation Metric:** ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
- **Target Variable:** `Will_Buy_EV` (Yes / No)
- **Submission Format:** Predicted probabilities (continuous values between 0 and 1) for each `id` in the test set

The ROC-AUC metric was chosen by the competition because it measures the model's ability to **rank** positive examples higher than negative ones, regardless of the classification threshold. This is especially important here because the dataset is imbalanced (~17% positive class), and accuracy alone would be misleading.

---

## Project Structure

```
electricVehicle/
├── data/
│   ├── train.csv                # Training data (668,665 rows x 15 cols)
│   ├── test.csv                 # Test data (286,571 rows x 14 cols)
│   └── sample_submission.csv    # Submission format reference
├── models/
│   ├── best_model.pkl           # Best performing model (Stacking Ensemble)
│   ├── catboost.pkl             # Trained CatBoost classifier
│   ├── decision_tree.pkl        # Trained Decision Tree classifier
│   ├── encoding_info.pkl        # Feature encoding metadata (maps, column lists)
│   ├── feature_columns.pkl      # Ordered list of feature column names after encoding
│   ├── knn.pkl                  # Trained K-Nearest Neighbors classifier
│   ├── lightgbm.pkl             # Trained LightGBM classifier
│   ├── logistic_regression.pkl  # Trained Logistic Regression classifier
│   ├── random_forest.pkl        # Trained Random Forest classifier (300 trees)
│   ├── scaler.pkl               # Fitted StandardScaler for distance-based models
│   ├── stacking_ensemble.pkl    # Trained Stacking Ensemble (XGB + LGB + CAT -> LR)
│   ├── svm_(sample).pkl         # Trained SVM on 50k subsample
│   ├── voting_ensemble.pkl      # Trained Soft Voting Ensemble
│   └── xgboost.pkl              # Trained XGBoost classifier
├── 2EDA.ipynb                   # Comprehensive Exploratory Data Analysis
├── EDA.ipynb                    # Initial EDA and rapid model prototyping
├── Training.ipynb               # Full model training, evaluation, and comparison
├── Predict.ipynb                # Test set inference and submission generation
├── submission.csv               # Final Kaggle submission file (286,571 predictions)
├── decision.md                  # Detailed decision log and methodology walkthrough
├── .gitignore
└── README.md
```

### Notebook Descriptions

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `2EDA.ipynb` | Deep-dive exploratory analysis with 30+ visualizations, statistical tests (chi-square, Cramer's V, VIF), pivot tables, and correlation analysis | Understanding of feature distributions, relationships, and data quality |
| `EDA.ipynb` | Quick initial exploration and rapid prototyping of baseline models to establish a performance floor | First-pass AUC scores for Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Voting, and Stacking |
| `Training.ipynb` | Systematic training of 10 models (8 individual + 2 ensemble) with standardized evaluation, comparison charts, ROC curves, confusion matrices, and cross-validation | All serialized model artifacts in `models/`, feature importance plots, model comparison table |
| `Predict.ipynb` | Loading the best model, transforming test data identically, generating predictions, validating submission format, and analyzing prediction distributions | `submission.csv`, prediction analysis visualizations, single-row prediction demo |

---

## Dataset Overview

### Features (13 input columns)

| Feature | Type | Description | Value Range / Categories |
|---------|------|-------------|--------------------------|
| `Age` | Numerical | Age of the consumer | 25 to 70 |
| `Annual_Income_USD` | Numerical | Yearly income in US dollars | Continuous |
| `Daily_Commute_km` | Numerical | Daily one-way travel distance in kilometers | Continuous |
| `Number_of_Cars_Owned` | Numerical | Total number of cars the consumer currently owns | Integer |
| `Charging_Stations_Near_Home` | Numerical | Count of EV charging stations within proximity of home | Integer |
| `Charging_Stations_Near_Work` | Numerical | Count of EV charging stations within proximity of workplace | Integer |
| `Environmental_Concern_Level` | Numerical | Self-reported environmental concern on a 1-to-5 scale | 1.0 to 5.0 |
| `Gender` | Categorical | Gender identity of the consumer | Male, Female, Other |
| `City_Type` | Categorical | Type of city the consumer resides in | Urban, Suburban, Rural |
| `Current_Car_Type` | Categorical | Category of the consumer's current vehicle | Sedan, SUV, Hatchback, Truck |
| `Home_Charging_Possible` | Binary | Whether the consumer can install a home charging station | Yes, No |
| `Subsidy_Available` | Binary | Whether government EV subsidy is available to the consumer | Yes, No |
| `Range_Anxiety_Level` | Ordinal | Consumer's concern about EV battery range | Low, Medium, High |

### Target Variable

| Label | Count | Percentage |
|-------|-------|------------|
| No | 551,886 | 82.6% |
| Yes | 116,779 | 17.4% |

The dataset exhibits a **class imbalance** with roughly a 4.7:1 ratio of negative to positive examples. This imbalance influenced the choice of evaluation metric (ROC-AUC rather than accuracy) and the use of stratified splitting throughout the pipeline to preserve class proportions.

### Dataset Scale

| Split | Rows | Columns | Notes |
|-------|------|---------|-------|
| Train | 668,665 | 15 | Includes `id` column and `Will_Buy_EV` target |
| Test | 286,571 | 14 | Includes `id` column, no target |
| Total | 955,236 | — | Nearly 1 million samples |

The dataset has **zero missing values** and **zero duplicate rows**, making it clean and ready for modeling without imputation.

---

## Exploratory Data Analysis

Detailed EDA is documented across two notebooks: `2EDA.ipynb` (comprehensive analysis) and `EDA.ipynb` (initial exploration). Together, they produce 30+ visualizations and statistical tests.

### Univariate Analysis
- **Numerical features:** Distribution plots (histograms with KDE overlays), box plots to identify outlier ranges, and descriptive statistics for all 7 numerical columns
- **Categorical features:** Value counts and bar charts for Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, and Range_Anxiety_Level
- **Target distribution:** Bar chart confirming the 82.6% / 17.4% imbalance

### Bivariate and Multivariate Analysis
- **Numerical vs. Target:** Overlapping histograms (density-normalized) and grouped box plots comparing feature distributions for `Will_Buy_EV = Yes` vs. `No`. This revealed that features like `Environmental_Concern_Level`, `Charging_Stations_Near_Home`, and `Charging_Stations_Near_Work` show meaningful separation between classes.
- **Categorical vs. Target:** Stacked bar charts showing the EV purchase rate within each category. For example, Urban consumers and those with home charging access showed higher purchase rates.
- **Pivot Tables:** Multi-dimensional aggregation tables including:
  - Mean income, commute, and charging access by City_Type crossed with Will_Buy_EV
  - Income statistics (mean, median, count) by Gender crossed with Range_Anxiety_Level
  - EV purchase rate (%) by Current_Car_Type crossed with Subsidy_Available
  - EV purchase rate heatmap by City_Type crossed with Current_Car_Type
- **Pair Plot:** 2,000-sample scatter matrix of key numerical features (Income, Commute, Charging Stations, Environmental Concern) colored by target class

### Statistical Analysis
- **Pearson Correlation Heatmap:** Lower-triangle heatmap of all numerical and encoded features. Identified positive correlations between charging station access and purchase likelihood, and negative correlation between range anxiety and purchase likelihood.
- **Feature-Target Correlation:** Horizontal bar chart ranking all features by their Pearson correlation with `Will_Buy_EV`, colored by direction (green for positive, red for negative)
- **Chi-Square Independence Tests:** Applied to all categorical features against the target, with p-values and degrees of freedom. All categorical features were statistically significant predictors.
- **Cramer's V Association Matrix:** Full inter-categorical association heatmap to measure the strength of association between every pair of categorical variables
- **Variance Inflation Factor (VIF):** Computed for all numerical features to detect multicollinearity. No features showed dangerously high VIF values requiring removal.

### Segmentation and Aggregation
- **City-type profiles:** Aggregated mean income, median income, mean commute distance, and mean charging station counts by city type
- **Car type x target profiles:** Mean income and environmental concern by car type, split by purchase decision
- **Age group analysis:** Consumers binned into age groups (25-30, 31-40, 41-50, 51-60, 61-70) with a dual-axis chart showing both the count of consumers and the EV purchase rate per bin

---

## Feature Engineering

The preprocessing pipeline transforms raw categorical features into numerical representations. The exact same transformation logic is applied during both training and inference to prevent train-test skew.

### Transformation Steps

```
Raw Data (13 features, mixed types)
  |
  |-- Step 1: Binary Encoding
  |     Home_Charging_Possible  -->  {Yes: 1, No: 0}
  |     Subsidy_Available       -->  {Yes: 1, No: 0}
  |
  |-- Step 2: Ordinal (Label) Encoding
  |     Range_Anxiety_Level     -->  {Low: 0, Medium: 1, High: 2}
  |
  |-- Step 3: One-Hot Encoding (no drop_first)
  |     Gender          -->  Gender_Male, Gender_Female, Gender_Other
  |     City_Type       -->  City_Type_Urban, City_Type_Suburban, City_Type_Rural
  |     Current_Car_Type -->  Current_Car_Type_Sedan, Current_Car_Type_SUV,
  |                           Current_Car_Type_Hatchback, Current_Car_Type_Truck
  |
  |-- Step 4: Boolean-to-Integer Conversion
  |     Pandas get_dummies produces bool dtype; convert to int for model compatibility
  |
  |-- Step 5: Column Alignment
  |     Ensure test DataFrame has exactly the same columns in the same order as training
  |     Missing columns are filled with 0; extra columns are dropped
  |
  Output: 17 numerical features, ready for modeling
```

### Artifact Persistence

All encoding metadata is serialized to disk so that the exact same transformations can be reproduced at inference time without access to the training notebook:

| Artifact | Contents |
|----------|----------|
| `models/encoding_info.pkl` | Dictionary containing binary column names, range anxiety mapping, and nominal column names |
| `models/feature_columns.pkl` | Ordered list of 17 feature column names after encoding |
| `models/scaler.pkl` | Fitted `StandardScaler` instance (used only for models that require feature scaling) |

---

## Model Training and Evaluation

### Training Setup

- **Train/Validation Split:** 70% train / 30% validation, stratified by target class
- **Random State:** 42 (fixed across all models for reproducibility)
- **Scaling:** `StandardScaler` applied only to models that require it (Logistic Regression, KNN, SVM). Tree-based models received unscaled features.
- **Evaluation Function:** A standardized `evaluate()` function that fits the model, predicts on the validation set, computes metrics (Accuracy, ROC-AUC, F1-Score), records training time, prints a full classification report, and serializes the model to disk.

### Individual Models

| No. | Model | Scaling | Key Hyperparameters | Notes |
|-----|-------|---------|---------------------|-------|
| 1 | Logistic Regression | Yes | `max_iter=1000` | Linear baseline; serves as meta-learner in stacking |
| 2 | Decision Tree | No | `max_depth=10, min_samples_split=20` | Interpretable but prone to overfitting |
| 3 | Random Forest | No | `n_estimators=300, max_depth=15` | Bagging ensemble; also used for feature importance analysis |
| 4 | XGBoost | No | `n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8` | Gradient boosting with regularization |
| 5 | LightGBM | No | `n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8` | Histogram-based gradient boosting; faster training |
| 6 | CatBoost | No | `iterations=500, learning_rate=0.05, depth=6` | Gradient boosting with ordered boosting |
| 7 | KNN | Yes | `n_neighbors=7` | Instance-based learning; requires scaled features |
| 8 | SVM (RBF kernel) | Yes | `kernel='rbf', probability=True` | Trained on a 50,000-row subsample due to computational cost |

### Ensemble Models

| No. | Model | Strategy | Base Learners | Meta-Learner |
|-----|-------|----------|---------------|--------------|
| 9 | Soft Voting Ensemble | Averaged predicted probabilities | XGBoost + LightGBM + CatBoost | None (simple average) |
| 10 | **Stacking Ensemble** | Out-of-fold predictions as meta-features, 5-fold CV | XGBoost + LightGBM + CatBoost | Logistic Regression (`max_iter=1000`) |

### Evaluation and Comparison

All 10 models were evaluated on the same 30% stratified holdout set. The `Training.ipynb` notebook produces:

- **Model comparison table:** Styled DataFrame ranking all models by ROC-AUC with background gradient coloring
- **Triple bar chart:** Side-by-side horizontal bar charts comparing ROC-AUC, Accuracy, and F1-Score across all models with annotated values
- **Overlaid ROC curves:** All individual models plotted on a single ROC space with AUC values in the legend and a random baseline reference line
- **Confusion matrices:** Top 4 models displayed side by side with labeled cells (True Positives, False Positives, True Negatives, False Negatives)
- **Feature importance:** Horizontal bar chart of Random Forest `feature_importances_` ranked by importance

### Cross-Validation

After identifying the best model, a **5-Fold Stratified Cross-Validation** was performed on a 100,000-row subsample to verify that the model generalizes consistently. The CV scores, along with their mean and standard deviation, are printed to confirm low variance across folds.

---

## Best Model — Stacking Ensemble

The Stacking Ensemble achieved the highest ROC-AUC on both the local validation set and the Kaggle leaderboard.

```
+---------------------------------------------+
|              Stacking Ensemble               |
+---------------------------------------------+
|  Base Learners (Level 0):                    |
|    +-- XGBoost   (500 trees, lr=0.05)        |
|    +-- LightGBM  (500 trees, lr=0.05)        |
|    +-- CatBoost  (500 iters, lr=0.05)        |
|                                              |
|  Meta-Learner (Level 1):                     |
|    +-- Logistic Regression (max_iter=1000)   |
|                                              |
|  CV Strategy: 5-Fold Stratified              |
+---------------------------------------------+
|  Kaggle Public Score:   0.94132              |
|  Kaggle Private Score:  0.94070              |
+---------------------------------------------+
```

### How Stacking Works

1. **Out-of-fold training:** The training data is split into 5 folds. For each fold, the three base learners are trained on 4 folds and generate predictions on the held-out fold. This produces out-of-fold predictions for the entire training set without data leakage.
2. **Meta-feature construction:** The out-of-fold predictions from each base learner (3 probability columns) become the input features for the meta-learner.
3. **Meta-learner training:** A Logistic Regression model is trained on these 3 meta-features to learn the optimal linear combination of base model outputs.
4. **Inference:** For new data, all 3 base learners generate predictions, and the meta-learner combines them into a single final probability.

### Why Stacking Outperformed Voting

Soft voting simply averages the probabilities from all base learners with equal weight. Stacking, on the other hand, **learns** the optimal weights through the meta-learner. If one base model is systematically better in certain regions of the feature space, stacking can exploit this, whereas voting cannot.

---

## Inference Pipeline

The full prediction workflow is implemented in `Predict.ipynb` and follows a reproducible, artifact-driven approach.

### Step-by-Step Flow

```
test.csv
  |
  +-- Load serialized artifacts
  |     best_model.pkl, encoding_info.pkl, feature_columns.pkl, scaler.pkl
  |
  +-- Extract and store test IDs, then drop the 'id' column
  |
  +-- Apply transform_data() function
  |     Binary encoding --> Ordinal encoding --> One-Hot encoding --> Column alignment
  |
  +-- Check if model requires feature scaling
  |     Auto-detected from model class name (LogisticRegression, KNeighborsClassifier, SVC)
  |     If yes: apply scaler.transform()
  |     If no: use raw features
  |
  +-- Generate predictions
  |     model.predict_proba(X_test)[:, 1]  -->  continuous probabilities
  |     model.predict(X_test)              -->  binary class labels (for analysis)
  |
  +-- Build submission DataFrame: {id, Will_Buy_EV (probability)}
  |
  +-- Validation assertions
  |     Row count matches sample_submission.csv
  |     Column names match expected format
  |     All probabilities are within [0, 1]
  |
  +-- Save to submission.csv
```

### Prediction Analysis

After generating predictions, the notebook produces several diagnostic visualizations:

- **Probability distribution histogram:** Shows the spread of predicted probabilities with a threshold line at 0.5
- **Class distribution bar chart:** Counts of predicted Yes vs. No with annotated totals
- **Category breakdowns:** Mean predicted probability by City Type, Gender, and Car Type
- **Box plots:** Distribution of predicted probabilities segmented by Home Charging, Subsidy Available, and Range Anxiety Level
- **Scatter plots:** Income vs. predicted probability and Charging Stations Near Home vs. predicted probability (sampled 5,000 points)

### Single-Row Prediction Demo

The notebook concludes with a demonstration of predicting for individual consumers:

- **High-affinity profile:** 35-year-old female, $95k income, urban, sedan, home charging available, subsidy available, low range anxiety — predicted high probability
- **Low-affinity profile:** 60-year-old male, $40k income, rural, truck, no home charging, no subsidy, high range anxiety — predicted low probability

A side-by-side horizontal bar chart compares the two predictions with a decision threshold reference line.

---

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib scipy statsmodels
```

### Reproduce the Full Pipeline

```bash
# Step 1 (Optional): Run the detailed exploratory data analysis
jupyter notebook 2EDA.ipynb

# Step 2: Train all 10 models, evaluate, compare, and save artifacts
jupyter notebook Training.ipynb

# Step 3: Generate test predictions and create the Kaggle submission file
jupyter notebook Predict.ipynb
# --> Outputs submission.csv (286,571 rows)
```

### Using the Trained Model Directly

If you want to load the pre-trained model and make predictions on new data without retraining:

```python
import joblib
import pandas as pd

# Load all necessary artifacts
model         = joblib.load('models/best_model.pkl')
encoding_info = joblib.load('models/encoding_info.pkl')
feature_cols  = joblib.load('models/feature_columns.pkl')
scaler        = joblib.load('models/scaler.pkl')

# Define the transformation function (same as in Predict.ipynb)
def transform_data(df, encoding_info, feature_cols):
    df = df.copy()

    # Binary encoding
    for col in encoding_info['binary_cols']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Ordinal encoding
    df['Range_Anxiety_Level'] = df['Range_Anxiety_Level'].map(
        encoding_info['range_anxiety_map']
    )

    # One-hot encoding
    df = pd.get_dummies(df, columns=encoding_info['nominal_cols'], drop_first=False)

    # Bool to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Align columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df

# Prepare your data
new_data = pd.DataFrame([{
    'Age': 35,
    'Annual_Income_USD': 95000.0,
    'Daily_Commute_km': 25.0,
    'Number_of_Cars_Owned': 1,
    'Charging_Stations_Near_Home': 8,
    'Charging_Stations_Near_Work': 12,
    'Environmental_Concern_Level': 4.0,
    'Gender': 'Female',
    'City_Type': 'Urban',
    'Current_Car_Type': 'Sedan',
    'Home_Charging_Possible': 'Yes',
    'Subsidy_Available': 'Yes',
    'Range_Anxiety_Level': 'Low'
}])

X_transformed = transform_data(new_data, encoding_info, feature_cols)

# Predict
probability = model.predict_proba(X_transformed)[:, 1]
print(f"Purchase probability: {probability[0]:.4f}")
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (pipelines, ensembles, metrics, preprocessing) |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Statistical Analysis | SciPy (chi-square tests), Statsmodels (VIF) |
| Model Serialization | Joblib |
| Development Environment | Jupyter Notebook |
| Competition Platform | Kaggle |

---

## Author

**Jatin Nath**

---
