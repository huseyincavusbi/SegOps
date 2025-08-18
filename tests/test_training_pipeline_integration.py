import os
import pandas as pd
import shutil
import subprocess

def test_training_pipeline_integration(tmp_path):
    # Copy sample data to a temp location
    sample_csv = os.path.join(os.path.dirname(__file__), 'test_data_sample.csv')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_csv = data_dir / "ncr_ride_bookings.csv"
    shutil.copy(sample_csv, test_csv)

    # Run the training script with the temp data dir
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/models/train_minibatch_kmeans.py"))
    result = subprocess.run([
        "python", script_path
    ], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check output file
    output_csv = tmp_path / "data" / "minibatch_kmeans_clusters.csv"
    assert output_csv.exists(), "Output CSV not created"
    df = pd.read_csv(output_csv)
    assert "cluster" in df.columns, "Cluster column missing in output"
    assert len(df) == 5, "Output row count mismatch"
