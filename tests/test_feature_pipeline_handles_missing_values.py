import pandas as pd
from src.features.feature_pipeline import build_feature_pipeline

def test_feature_pipeline_handles_missing_values():
    df = pd.DataFrame({
        'Booking Value': [100, None],
        'Ride Distance': [5, 10],
        'Driver Ratings': [4.5, 4.0],
        'Customer Rating': [5, 4],
        'Vehicle Type': ['Sedan', None],
        'Payment Method': ['Card', 'Cash']
    })
    numeric = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
    categorical = ['Vehicle Type', 'Payment Method']
    pipeline = build_feature_pipeline(numeric, categorical)
    # Fill numeric columns with 0, categorical with 'missing'
    df_filled = df.copy()
    for col in numeric:
        df_filled[col] = df_filled[col].fillna(0)
    for col in categorical:
        df_filled[col] = df_filled[col].fillna('missing')
    try:
        X = pipeline.fit_transform(df_filled)
        assert X.shape[0] == 2
    except Exception as e:
        assert False, f"Pipeline failed on missing values: {e}"
