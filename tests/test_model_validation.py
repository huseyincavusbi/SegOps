
import os
import pandas as pd
import shutil
import subprocess
from sklearn.metrics import silhouette_score
from src.features.feature_pipeline import build_feature_pipeline

def test_model_silhouette_score(tmp_path):
    # Copy test data to temp dir
    sample_csv = os.path.join(os.path.dirname(__file__), 'test_data_sample.csv')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_csv = data_dir / "ncr_ride_bookings.csv"
    shutil.copy(sample_csv, test_csv)

    # Run the training script in temp dir
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/models/train_minibatch_kmeans.py"))
    result = subprocess.run([
        "python", script_path
    ], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Load test data and output
    df = pd.read_csv(test_csv)
    numeric = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
    categorical = ['Vehicle Type', 'Payment Method']
    pipeline = build_feature_pipeline(numeric, categorical)
    X = pipeline.fit_transform(df)
    output_csv = tmp_path / "data" / "minibatch_kmeans_clusters.csv"
    assert output_csv.exists(), "Output CSV not created"
    df_out = pd.read_csv(output_csv)
    assert 'cluster' in df_out.columns, "Cluster column missing in output"
    # Calculate silhouette score
    score = silhouette_score(X, df_out['cluster'])
    assert isinstance(score, float), f"Silhouette score not computed: {score}"
