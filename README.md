
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
├── data/           # Raw, processed data, and model artifacts (DVC tracked)
├── notebooks/      # EDA, prototyping (Jupyter)
├── src/
│   ├── api/        # FastAPI app (serve_clusters.py)
│   ├── features/   # Feature engineering pipeline
│   ├── models/     # Model training scripts
│   └── utils/      # Utilities
├── tests/          # Unit/integration/model validation tests
├── scripts/        # Automation scripts (e.g., batch prediction)
├── Dockerfile      # Multi-arch compatible
├── docker-compose.yml
├── requirements.txt
├── prometheus.yml
├── README.md
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
