
#!/bin/bash
# Batch Prediction Automation Script
# ----------------------------------
# This script runs the customer segmentation pipeline in batch mode.
# To automate, schedule this script with cron (system scheduler):
# Example (run daily at midnight):
# 0 0 * * * /path/to/the/project/scripts/run_batch_prediction.sh
# Make sure this script is executable: chmod +x scripts/run_batch_prediction.sh
#
# You do NOT need to change anything in your project folders to use cron.
# Cron will call this script at the scheduled time.

cd "$(dirname "$0")/.."

# Activate virtual environment if needed
if [ -f "../../.venv/bin/activate" ]; then
    source ../../.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the batch clustering pipeline
PYTHONPATH=. python src/models/train_minibatch_kmeans.py

echo "Batch prediction complete. Results saved to data/minibatch_kmeans_clusters.csv"
