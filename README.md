# Loan Default Prediction

## Project Overview
This project aims to predict loan defaults using an anonymized dataset. 
The goal is to build a professional ML pipeline following best practices.

## Project Structure

loan-default-ml/
├── app/                  # Scripts or API (future)
├── data/                 # Raw and processed datasets (ignored by Git)
├── models/               # Trained models (future)
├── notebooks/            # Notebooks for analysis and experiments
├── requirements.txt      # Python dependencies
├── README.md             # Project description
├── .gitignore            # Files/folders ignored by Git

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

## Current Status
- EDA: completed
- Preprocessing : completed
- Build Badeline Models: completed
- Model Evaluation: upcoming