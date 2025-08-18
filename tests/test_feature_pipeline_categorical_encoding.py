import pandas as pd
from src.features.feature_pipeline import build_feature_pipeline

def test_feature_pipeline_categorical_encoding():
    df = pd.DataFrame({
        'Booking Value': [100, 200],
        'Ride Distance': [5, 10],
        'Driver Ratings': [4.5, 4.0],
        'Customer Rating': [5, 4],
        'Vehicle Type': ['Sedan', 'SUV'],
        'Payment Method': ['Card', 'Cash']
    })
    numeric = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
    categorical = ['Vehicle Type', 'Payment Method']
    pipeline = build_feature_pipeline(numeric, categorical)
    X = pipeline.fit_transform(df)
    # Should have more columns than just numeric features (due to one-hot encoding)
    assert X.shape[1] > len(numeric)
