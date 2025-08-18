import pandas as pd
from src.features.feature_pipeline import build_feature_pipeline

def test_feature_pipeline_output_shape_and_type():
    df = pd.DataFrame({
        'Booking Value': [100, 200, 150],
        'Ride Distance': [5, 10, 7],
        'Driver Ratings': [4.5, 4.0, 4.2],
        'Customer Rating': [5, 4, 4],
        'Vehicle Type': ['Sedan', 'SUV', 'Sedan'],
        'Payment Method': ['Card', 'Cash', 'Card']
    })
    numeric = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
    categorical = ['Vehicle Type', 'Payment Method']
    pipeline = build_feature_pipeline(numeric, categorical)
    X = pipeline.fit_transform(df)
    assert X.shape[0] == 3
    # Output should be numpy array or sparse matrix
    import numpy as np
    from scipy import sparse
    assert isinstance(X, (np.ndarray, sparse.spmatrix))
