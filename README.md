
# SegOps: Customer Segmentation with MLOps/DevOps

SegOps is a production-grade platform for customer segmentation, analytics, and MLOps/DevOps best practices. It enables scalable, reproducible machine learning workflows, robust monitoring, and rapid deployment for real-world segmentation use cases.

## Features
- **Data**: The project uses Uber ride booking data (e.g., `ncr_ride_bookings.csv`) containing anonymized trip records, timestamps, locations, and customer attributes. This data is used for exploratory analysis, feature engineering, and clustering. All data is versioned with DVC for reproducibility. See the `data/` directory for details. 
	- Source: [Kaggle - Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- Automated customer segmentation using MiniBatch KMeans
- Modular, reusable feature engineering pipeline
- REST API (FastAPI) for real-time inference
- Batch prediction scripts
- Full MLOps/DevOps stack: experiment tracking, CI/CD, containerization, monitoring
- Data and model versioning

## Project Structure
```
├── .dvc/                        # DVC metadata and cache
├── .dvcignore                   # DVC ignore file
├── .git/                        # Git metadata
├── .github/                     # GitHub Actions/workflows
│   └── workflows/
├── .gitignore
├── .pytest_cache/               # pytest cache
├── Dockerfile                   # Multi-arch Docker build
├── README.md
├── data/                        # Data and model artifacts
│   ├── chunks_head/
│   ├── feature_pipeline.joblib
│   ├── minibatch_kmeans_clusters.csv
│   ├── minibatch_kmeans_model.joblib
│   ├── ncr_ride_bookings.csv
│   ├── ncr_ride_bookings.csv.dvc
│   ├── queries.active
│   └── wal/
├── docker-compose.yml
├── notebooks/                   # Jupyter notebooks
│   ├── cluster_profiling.ipynb
│   ├── data_validation_gx.ipynb
│   └── uber_eda.ipynb
├── prometheus.yml
├── requirements.txt
├── scripts/                     # Automation scripts
│   └── run_batch_prediction.sh
├── setup.py
├── src/                         # Source code
│   ├── api/
│   ├── features/
│   └── models/
├── tests/                       # Unit/integration/model validation tests
│   ├── __pycache__/
│   ├── test_api.py
│   ├── test_data_sample.csv
│   ├── test_feature_pipeline.py
│   ├── test_feature_pipeline_all_expected_features.py
│   ├── test_feature_pipeline_categorical_encoding.py
│   ├── test_feature_pipeline_handles_missing_values.py
│   ├── test_feature_pipeline_output_shape_and_type.py
│   ├── test_model_validation.py
│   └── test_training_pipeline_integration.py
```

## Tech Stack & Tools
- **Python 3.11**, pandas, scikit-learn, joblib
- **FastAPI** (REST API)
- **DVC** (data/model versioning)
- **MLflow** (experiment tracking)
- **pytest** (testing)
- **Docker & Docker Compose** (multi-arch containerization)
- **Prometheus & Grafana** (monitoring & dashboards)
- **GitHub Actions** (CI/CD)

## Quickstart
### Docker Image
Multi-architecture image: [huseyincavus/segops on Docker Hub](https://hub.docker.com/r/huseyincavus/segops)


### 1. Local Development
```sh
git clone https://github.com/huseyincavusbi/SegOps.git
cd SegOps
pip install -r requirements.txt
# Run API locally
uvicorn src.api.serve_clusters:app --reload
```

### 2. Run with Docker Compose (Recommended)
```sh
docker-compose up --build
# Access API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### 3. Build Multi-Arch Docker Image
```sh
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/segops:latest --push .
```

## License
MIT License. See `LICENSE` for details.
