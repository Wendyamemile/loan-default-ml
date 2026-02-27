# Loan Default Prediction

## Project Overview
This project aims to predict loan defaults using an anonymized dataset. 
The goal is to build a professional ML pipeline following best practices.

## Current Status
- EDA: completed
- Preprocessing : completed
- Build Badeline Models: completed
- Model Evaluation: completed
- Improvement: completed
- Future improvement ( Advance models ) : upcoming

## How to Run
- Clone the repository
```bash
git clone https://github.com/Wendyamemile/loan-default-ml.git
```

## Project Structure
```bash
loan-default-ml/
│
├── app/                # Scripts or API (future)
├── data/               # Raw and processed datasets (ignored by Git)
│   ├── raw/
│   └── processed/
├── models/             # Trained models (future)
├── notebooks/          # Analysis and experiments
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Files ignored by Git
```

## Dataset
The dataset used in this project comes from kaggle and is not included in this repository.

To reproduce the projects:
1. Download the dataset from kaggle
2. Place the CSV file in 'data/raw'

## Config
- src/config.py is used to store constants for the project, such as:
  - TARGET variable (`Default`)
  - Numerical features list
  - Categorical features list

## EDA 
The exploratory data analysis (EDA) is in 'notebooks/01_eda.ipynb'.
- Target variables analysis (Default (imbalanced: ~11.7%))
- Feature distributions
- Missing values check
- Correlations betwen features
- ROC-AUC ( chosen due to class imbalance )

## Preprocessing 
This step transform raw data into a format suitable for modeling. All features are prepared using pipelines for consistency an reusability. It is in 'notebooks/01_preprocessing.ipynb'. 

### Preprocessing steps
- Load dataset
- Separate features and target
    - Features: 'NUM_FEATURES' + 'CAT_FEATURES'
    - Target: 'TARGET' 
- Train/Test split
    - 80/20 split ('TEST_SIZE=0.2')
    - Stratified to preserve class imbalance ('RANDOM_STATE=42')
- Numeric features preprocessing - scale with 'StandardScaler'
- Categorical features preprocessing - encoded with 'OneHotEncoder'
- Pipeline - combine numerical + categorical pipeline using 'ColumnTansformer'

### Preprocessing Module
- 'src/preprocessing.py' contains ** reusable preprocessing pipelines ** for consistent application in notebooks and modeling scripts.

### key takeaways
- Numeric features scaled, categorical features encoded
- Train/Test split preserves imbalanced target (~11,7%)
- Preprocessor ready for baseline modeling

## Build Baseline Models

The baseline model is the first, simple model in this project.  
It serves as a reference point for evaluating the performance of more complex models.

### Purpose
- Provides a performance floor to compare future models.  
- Helps check if features contain useful information.  
- Quick to implement and interpret.

### Implementation

- Classification - Random Forest
- Regression - Logistic Regression

## Model Evaluation
The evaluation of our initial model includes Logistic Regression (LR) and Random Forest (RF).
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization to check misclassifications
- Comparaison betwen models to see improvement
- Identification of the best-performing model for further tuning

## Show Results - Baseline Models
```bash
| Model                  | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------------|---------|-----------|--------|----------|---------|
| Logistic Regression    | 0.63    | 0.84      | 0.63   | 0.69     | 0.7001  |
| Random Forest          | 0.8830  | 0.8362    | 0.8830 | 0.8362   | 0.6557  |
```

Observations:
- Logistic Regression is interpretable and better at ranking positives (higher ROC-AUC).
- Random Forest has higher accuracy and recall but lower ROC-AUC → may overfit to the majority class.
- ROC-AUC is important for imbalanced data, showing how well the model separates defaults from non-defaults.

## Improved Model (RF + SMOTE + Hyperparameter Tuning + Threshold Tuning)


### Baseline Random Forest (original, no SMOTE)
- Accuracy: 0.8830
- Precision (class 1): 0.45
- Recall (class 1): 0.034
- F1-score (class 1): 0.06
- ROC-AUC: 0.6557
- Threshold: default (0.5)

Confusion Matrix (Baseline):
```bash
|          | Predicted 0 | Predicted 1 |
|----------|-------------|-------------|
| Actual 0 | 44888       | 251         |
| Actual 1 | 5725        | 206         |
```
---

### Improved Random Forest ( Hyperparameter Tuning + SMOTE + Threshold Tuning)
- Accuracy: 0.7687
- Precision (class 1): 0.24
- Recall (class 1): 0.45
- F1-score (class 1): 0.31
- ROC-AUC: 0.7002
- Threshold: ~ 0.5747

Confusion Matrix (Improved RF):
```bash
|          | Predicted 0 | Predicted 1 |
|----------|-------------|-------------|
| Actual 0 | 36602       | 8537        |
| Actual 1 | 3275        | 2656        |
```
> Note: Class 1 corresponds to loan defaults. SMOTE and threshold tuning improve detection of defaults (recall) at the cost of slightly lower overall accuracy.

Observation:  
- Improved pipeline detects many more actual defaults (TP increases from 206 → 2,656).  
- False positives increase (FP rises from 251 → 8,537), which reduces overall accuracy.  
- ROC-AUC improves (0.6557 → 0.7002), showing better separation between default and non-default.  
- Threshold tuning ensures optimal balance between precision and recall for real-world use.

---

## Summary

- Baseline Model: Achieved high overall accuracy but failed to detect most default cases due to class imbalance.
- Improved Model (SMOTE + Tuning + Threshold Optimization): Significantly improved default detection, increased ROC-AUC, and achieved a more balanced F1-score.
- Trade-off: Slight decrease in overall accuracy, but substantially better identification of high-risk borrowers.

Final Choice: The improved Random Forest model was selected because it better aligns with the business objective of minimizing credit risk by detecting potential loan defaults.


