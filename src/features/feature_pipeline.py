from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_feature_pipeline(numeric_features, categorical_features):
    """
    Build a reusable feature engineering pipeline for numeric and categorical features.
    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
    Returns:
        sklearn ColumnTransformer: Feature transformation pipeline.
    """
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor
