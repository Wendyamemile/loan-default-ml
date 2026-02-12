from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(num_features, cat_features):

    # Numerical Pipeline
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES)
    ])

    # Return a fitted ColumnTransformer for preprocessing
    return preprocessor