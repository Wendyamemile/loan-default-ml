from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_lr_pipeline(preprocessor=None, random_state=None):
    """
    Build Logistic Regression pipeline.
    Parameters:
    - preprocessor: optional preprocessing step (or None if already preprocessed)
    - random_state: pass from config.py
    Returns:
    - Pipeline
    """
    steps = []
    if preprocessor:
        steps.append(('preprocessor', preprocessor))
    steps.append(('classifier', LogisticRegression(class_weight='balanced',
                                                   random_state=random_state)))
    return Pipeline(steps)

def build_rf_pipeline(preprocessor=None, random_state=None, n_estimators=100):
    """
    Build Random Forest pipeline.
    Parameters:
    - preprocessor: optional preprocessing step
    - random_state: pass from config.py
    """
    steps = []
    if preprocessor:
        steps.append(('preprocessor', preprocessor))
    steps.append(('classifier', RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=random_state)))
    return Pipeline(steps)