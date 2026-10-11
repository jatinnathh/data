# Decision Log — Electric Vehicle Purchase Prediction

This document records every major decision made during the project, the reasoning behind each choice, and the step-by-step workflow followed from raw data to Kaggle submission.

---

## Step 1: Understanding the Problem

**What I did:**
- Read the competition description and understood that this is a binary classification problem
- The task is to predict whether a consumer will buy an Electric Vehicle (`Will_Buy_EV`: Yes/No)
- The evaluation metric is **ROC-AUC**, which measures the model's ability to rank positive cases higher than negative ones
- The submission requires **probabilities** (not class labels), so `predict_proba()` is needed, not `predict()`

**Key insight:** Since the metric is ROC-AUC, optimizing for raw accuracy would be a mistake. ROC-AUC rewards correct probability ordering, so well-calibrated probability estimates matter more than hard classification boundaries.

---

## Step 2: Initial Data Exploration (EDA.ipynb)

**What I did:**
- Loaded `train.csv` (668,665 rows, 15 columns including `id` and target)
- Dropped the `id` column since it carries no predictive information
- Examined the target distribution: 82.6% No, 17.4% Yes — confirmed class imbalance
- Identified column types:
  - 7 numerical features (Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned, Charging_Stations_Near_Home, Charging_Stations_Near_Work, Environmental_Concern_Level)
  - 6 categorical/binary/ordinal features (Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level)
- Checked for missing values: **none found**
- Checked for duplicates: **none found**

**Decision:** No imputation or deduplication required. The data is clean and ready for feature encoding.

---

## Step 3: Feature Encoding Decisions

**What I did and why:**

### Binary columns → Map to 0/1
- `Home_Charging_Possible` and `Subsidy_Available` have exactly two values (Yes/No)
- **Decision:** Simple `.map({'Yes': 1, 'No': 0})` — straightforward and preserves interpretability
- **Why not one-hot?** With only 2 levels, one-hot would create two perfectly correlated columns. A single 0/1 column is sufficient.

### Ordinal column → Label encode with meaningful ordering
- `Range_Anxiety_Level` has three values: Low, Medium, High
- **Decision:** Map to `{Low: 0, Medium: 1, High: 2}` to preserve the natural ordering
- **Why not one-hot?** The levels have an inherent order (Low < Medium < High), and label encoding captures this ordinal relationship. One-hot encoding would discard this ordering information.

### Nominal columns → One-Hot Encoding (no drop_first)
- `Gender` (Male, Female, Other), `City_Type` (Urban, Suburban, Rural), `Current_Car_Type` (Sedan, SUV, Hatchback, Truck)
- **Decision:** Used `pd.get_dummies()` with `drop_first=False`
- **Why drop_first=False?** Tree-based models (which are the primary models here) don't suffer from multicollinearity. Keeping all dummy columns gives each tree split a direct feature to work with, rather than requiring the model to infer the dropped category from the absence of others.

### Boolean-to-Integer conversion
- Pandas `get_dummies()` creates boolean dtype columns
- **Decision:** Convert to `int` explicitly because some models (especially CatBoost) can behave unexpectedly with boolean dtypes

### Column alignment for inference
- **Decision:** At inference time, after one-hot encoding the test set, align columns to exactly match the training feature order. Fill any missing columns with 0, drop any extra columns.
- **Why?** If the test set happens to have different categorical values or different ordering, the model would receive misaligned features. This step guarantees consistency.

**Result:** 13 raw features transformed into **17 numerical features** ready for modeling.

---

## Step 4: Deep Exploratory Data Analysis (2EDA.ipynb)

**What I did:**

### Univariate Analysis
- Plotted histograms for all numerical features to understand distributions
- Plotted bar charts for all categorical features to see category frequencies
- **Finding:** The data is fairly well-distributed without extreme skew in most features

### Bivariate Analysis — Numerical vs. Target
- Overlapping density histograms and box plots for each numerical feature, split by `Will_Buy_EV`
- **Finding:** `Environmental_Concern_Level`, `Charging_Stations_Near_Home`, and `Charging_Stations_Near_Work` showed the clearest separation between buyers and non-buyers. Higher values in these features correlated with higher purchase likelihood.

### Bivariate Analysis — Categorical vs. Target
- Stacked bar charts showing the EV purchase rate within each category
- **Finding:** Urban consumers had higher purchase rates than Rural. Home charging availability was a strong differentiator.

### Pivot Table Analysis
- Created multi-dimensional aggregation tables:
  - Mean features by City_Type x Will_Buy_EV
  - Income stats by Gender x Range_Anxiety_Level
  - EV purchase rate (%) by Car_Type x Subsidy_Available
  - Purchase rate heatmap by City_Type x Current_Car_Type
- **Finding:** Subsidy availability consistently boosted purchase rates across all car types. The effect was strongest in Rural areas.

### Correlation Analysis
- Pearson correlation heatmap (lower triangle) for all numerical + encoded features
- Feature-target correlation bar chart
- **Finding:** `Environmental_Concern_Level` and charging station counts had the strongest positive correlations with the target. `Range_Anxiety_Level` had a negative correlation.

### Statistical Tests
- Chi-square independence tests for all categorical features vs. target
- **Finding:** All categorical features were statistically significant (p < 0.001). Cramer's V values indicated moderate association strength.
- Computed Cramer's V matrix for inter-categorical relationships
- **Finding:** No strong collinearity between categorical features

### Multicollinearity Check (VIF)
- Computed Variance Inflation Factors for all numerical features
- **Finding:** No features had dangerously high VIF values (all below commonly used thresholds). No features needed to be removed.

### Age Group Analysis
- Binned ages into 5 groups and computed purchase rates per bin
- **Finding:** Purchase rates were relatively stable across age groups, suggesting age alone is not a dominant predictor

**Decision:** All 13 features are kept. No feature removal necessary based on VIF or correlation analysis. The features provide complementary information.

---

## Step 5: Train/Validation Split

**What I did:**
- Split the training data 70/30 using `train_test_split` with `stratify=y` and `random_state=42`
- **Decision:** 70/30 split (not 80/20) to have a larger validation set for more reliable evaluation, given the dataset is large enough (668k rows) that the training set (467k) is still substantial
- **Decision:** Stratified split to preserve the 82.6/17.4 class ratio in both splits, preventing any split from being unrepresentative
- **Decision:** Fixed `random_state=42` across all operations for full reproducibility

**Also did:**
- Fitted a `StandardScaler` on the training split and saved it
- Applied `scaler.transform()` (not `fit_transform()`) on the validation split to prevent data leakage

---

## Step 6: Model Selection and Training (Training.ipynb)

I trained 8 individual models and 2 ensemble models, deliberately spanning different algorithm families to find the best approach.

### Model 1: Logistic Regression (Baseline)
- **Why:** Every classification project needs a linear baseline. If gradient boosting only barely beats logistic regression, the problem might be near-linear and not worth the complexity.
- **Configuration:** `max_iter=1000` to ensure convergence on a large dataset
- **Used scaled features** (logistic regression is sensitive to feature magnitude)
- **Result:** Established the performance floor

### Model 2: Decision Tree
- **Why:** Highly interpretable; helps understand which features and splits are most important. Also serves as a building block to compare against its ensemble variants.
- **Configuration:** `max_depth=10, min_samples_split=20` to prevent overfitting
- **Used unscaled features** (trees are invariant to feature scaling)
- **Result:** Moderate performance, as expected for a single tree

### Model 3: Random Forest
- **Why:** Bagging ensemble of decision trees that reduces variance. Also provides `feature_importances_` for model interpretability.
- **Configuration:** `n_estimators=300, max_depth=15` — enough trees for stable averaging, with limited depth to balance bias-variance
- **Result:** Significant improvement over single decision tree. Feature importance chart generated.
- **Feature importance finding:** Used the resulting feature importance plot to confirm that `Environmental_Concern_Level`, `Charging_Stations_Near_Home`, `Annual_Income_USD`, and `Charging_Stations_Near_Work` are the most important predictors

### Model 4: XGBoost
- **Why:** State-of-the-art gradient boosting. Regularized (L1/L2), handles missing values natively, and is highly competitive on tabular data.
- **Configuration:**
  - `n_estimators=500` — more boosting rounds since learning rate is low
  - `learning_rate=0.05` — slow learning for better generalization
  - `max_depth=6` — moderate depth; deeper than Random Forest individual trees because boosting corrects errors sequentially
  - `subsample=0.8, colsample_bytree=0.8` — row and column subsampling for regularization
  - `eval_metric='auc'` — optimize directly for the competition metric
- **Result:** Strong AUC, one of the top individual models

### Model 5: LightGBM
- **Why:** Histogram-based gradient boosting that is faster than XGBoost for large datasets. Uses leaf-wise tree growth instead of depth-wise, which can capture more complex patterns.
- **Configuration:**
  - `n_estimators=500, learning_rate=0.05` — same budget as XGBoost for fair comparison
  - `num_leaves=63` — leaf-wise growth parameter; 63 is roughly equivalent to max_depth=6 in depth-wise growth (2^6 - 1 = 63)
  - `subsample=0.8, colsample_bytree=0.8` — same regularization
  - `verbose=-1` — suppress training logs
- **Result:** Very competitive with XGBoost, faster training time

### Model 6: CatBoost
- **Why:** Gradient boosting with ordered boosting (uses random permutations to prevent target leakage during training). Also handles categorical features natively, though I pre-encoded them.
- **Configuration:** `iterations=500, learning_rate=0.05, depth=6, verbose=0`
- **Result:** Competitive with XGBoost and LightGBM; slightly different error patterns due to ordered boosting

### Model 7: KNN (K-Nearest Neighbors)
- **Why:** Non-parametric, instance-based method. Sometimes captures local patterns that tree-based models miss. Included for diversity.
- **Configuration:** `n_neighbors=7`
- **Used scaled features** (KNN uses distance measures that are sensitive to feature magnitude)
- **Result:** Lower AUC than boosting methods, but provides a different type of model

### Model 8: SVM (Support Vector Machine with RBF kernel)
- **Why:** Kernel-based method that can capture non-linear decision boundaries. Included for completeness.
- **Configuration:** `kernel='rbf', probability=True`
- **Critical decision:** Trained on only **50,000 rows** (random subsample) because SVM has O(n^2) to O(n^3) time complexity and training on 467k rows would be prohibitively slow
- **Used scaled features** (SVM is sensitive to feature magnitude)
- **Result:** Moderate performance; limited by the reduced training set

---

## Step 7: Model Comparison and Analysis

**What I did:**
- Compiled all model results into a comparison DataFrame sorted by ROC-AUC
- Created three visualizations:
  1. **Triple bar chart:** ROC-AUC, Accuracy, and F1-Score side by side for all models
  2. **Overlaid ROC curves:** All individual models on one plot with AUC in legend
  3. **Confusion matrices:** Top 4 models displayed with cell counts

**Key observations:**
- The three gradient boosting models (XGBoost, LightGBM, CatBoost) all performed very similarly — they occupied the top individual model positions
- Random Forest was close behind but slightly lower
- Logistic Regression provided a reasonable baseline but couldn't capture non-linear patterns
- Decision Tree and KNN were significantly weaker
- SVM was limited by the reduced training set

**Decision:** The top 3 gradient boosting models are the best candidates for ensembling. They use different algorithms (different loss functions, tree growth strategies, and regularization approaches) but achieve similar performance, making them ideal for combining — diverse models that are individually strong.

---

## Step 8: Ensemble Construction

### Soft Voting Ensemble
- **What:** Averaged the predicted probabilities from XGBoost, LightGBM, and CatBoost with equal weights
- **Why:** Simple, no additional training required, and averaging often smooths out individual model errors
- **Result:** Slightly better than any individual model

### Stacking Ensemble
- **What:** Used XGBoost, LightGBM, and CatBoost as base learners. Generated out-of-fold predictions via 5-fold stratified CV. Trained a Logistic Regression meta-learner on these predictions.
- **Why:** Unlike voting which uses fixed equal weights, stacking **learns** the optimal combination. If one model is more reliable in certain prediction ranges, the meta-learner can weight it accordingly.
- **Why Logistic Regression as meta-learner?** It's simple, fast, and unlikely to overfit on just 3 meta-features. A complex meta-learner would risk overfitting the small meta-feature space.
- **Why 5-fold CV for out-of-fold predictions?** Using direct training predictions would cause data leakage (the meta-learner would learn from predictions on data the base models were trained on). Out-of-fold predictions ensure the meta-learner sees genuine generalization performance.
- **Result:** Best AUC among all models — selected as the final model

**Decision:** Stacking Ensemble is the best model. Saved as `models/best_model.pkl`.

---

## Step 9: Cross-Validation Sanity Check

**What I did:**
- Ran 5-Fold Stratified Cross-Validation on a 100,000-row subsample using the best model
- **Why subsample?** The stacking ensemble is expensive to train (it trains 3 models x 5 folds = 15 model fits internally). Running full CV on 668k rows would be very slow. A 100k subsample is representative enough to verify stability.
- **Why stratified?** To preserve class ratios in each fold, consistent with the train/val split strategy

**Result:** CV scores were consistent across folds with low standard deviation, confirming the model generalizes well and the validation score is not a fluke.

---

## Step 10: Inference on Test Set (Predict.ipynb)

**What I did:**
1. Loaded all serialized artifacts: `best_model.pkl`, `encoding_info.pkl`, `feature_columns.pkl`, `scaler.pkl`
2. Loaded `test.csv` (286,571 rows), saved the `id` column separately, and dropped it from features
3. Applied the exact same `transform_data()` function used during training
4. Checked if the best model requires scaling (auto-detected by checking the model's class name)
5. Generated predicted probabilities using `predict_proba()[:, 1]`
6. Built the submission DataFrame with columns `{id, Will_Buy_EV}`
7. Ran assertion checks:
   - Row count matches `sample_submission.csv`
   - Column names match expected format
   - All predicted probabilities are within [0, 1]
8. Saved to `submission.csv`

**Key decision:** Submitted **probabilities** (continuous 0-1 values), not class labels (0 or 1). This is critical because the evaluation metric is ROC-AUC, which operates on ranked probabilities. Submitting hard labels would destroy the ranking information.

---

## Step 11: Prediction Analysis and Validation

**What I did:**
- Plotted the distribution of predicted probabilities — confirmed a reasonable spread (not all clustered at 0 or 1)
- Checked predicted class distribution (using 0.5 threshold) — confirmed it roughly matches the training set class ratio
- Analyzed predictions by demographic segments:
  - Urban consumers had higher predicted probabilities (consistent with EDA findings)
  - Home charging availability strongly influenced predictions
  - High range anxiety correlated with lower predictions
- Created scatter plots of Income vs. Probability and Charging Access vs. Probability
- Ran a single-row prediction demo with two contrasting consumer profiles to verify the model behaves intuitively

**Decision:** Predictions look reasonable and consistent with EDA insights. No signs of anomalies or data leakage.

---

## Step 12: Kaggle Submission

**What I did:**
- Uploaded `submission.csv` to the Kaggle competition page

**Result:**
| Metric | Score |
|--------|-------|
| Public Leaderboard ROC-AUC | **0.94132** |
| Private Leaderboard ROC-AUC | **0.94070** |

The minimal gap between public and private scores (0.00062) confirms that the model is not overfit to the public test set. The stacking ensemble generalized well.

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature encoding | Binary map + Ordinal label + One-Hot (no drop_first) | Preserves ordering for ordinal, avoids multicollinearity issues irrelevant to trees |
| Train/Val split | 70/30 stratified | Large enough dataset; bigger validation set for reliable evaluation |
| Scaling | Only for LR, KNN, SVM | Tree-based models are scale-invariant |
| SVM training size | 50k subsample | O(n^2-n^3) complexity makes full training infeasible |
| Ensemble base learners | XGBoost + LightGBM + CatBoost | Top 3 individual performers with algorithm diversity |
| Ensemble strategy | Stacking over Voting | Learned combination outperforms fixed averaging |
| Meta-learner | Logistic Regression | Simple, low risk of overfitting on 3 meta-features |
| Stacking CV | 5-fold stratified | Prevents data leakage in meta-feature generation |
| Submission values | Probabilities, not class labels | ROC-AUC requires continuous probability ranking |
| Cross-validation | 100k subsample, 5-fold | Validates generalization without excessive compute |

---
