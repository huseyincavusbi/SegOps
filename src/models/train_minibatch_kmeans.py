import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from src.features.feature_pipeline import build_feature_pipeline
import mlflow
import mlflow.sklearn

mlflow.set_experiment("uber-customer-segmentation")

# Load data
DATA_PATH = 'data/ncr_ride_bookings.csv'
df = pd.read_csv(DATA_PATH)

# Select features (customize as needed)
# Recommended features for customer segmentation
numeric_features = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']
categorical_features = ['Vehicle Type', 'Payment Method']


# Drop rows with nulls in selected features
selected_features = numeric_features + categorical_features
df_clean = df.dropna(subset=selected_features)


# Use the entire cleaned dataset for clustering
df_sample = df_clean

# Build and fit the feature pipeline
pipeline = build_feature_pipeline(numeric_features, categorical_features)
X = pipeline.fit_transform(df_sample)      


# Fit MiniBatchKMeans (k=5, based on silhouette score) and track with MLflow
k = 5
import joblib

with mlflow.start_run():
	kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
	kmeans.fit(X)
	# Assign cluster labels to the sample
	df_sample['cluster'] = kmeans.labels_
	# Calculate silhouette score
	sil_score = silhouette_score(X, kmeans.labels_)
	# Log parameters and metrics
	mlflow.log_param('k', k)
	mlflow.log_metric('silhouette_score', sil_score)
	# Log model artifact
	mlflow.sklearn.log_model(kmeans, "minibatch_kmeans_model")
	# Save results
	RESULT_PATH = 'data/minibatch_kmeans_clusters.csv'
	df_sample.to_csv(RESULT_PATH, index=False)
	mlflow.log_artifact(RESULT_PATH)
	print(f'Clustered sample saved to {RESULT_PATH}')

	# Save pipeline and model for API serving
	joblib.dump(pipeline, 'data/feature_pipeline.joblib')
	joblib.dump(kmeans, 'data/minibatch_kmeans_model.joblib')
	print('Saved pipeline to data/feature_pipeline.joblib')
	print('Saved model to data/minibatch_kmeans_model.joblib')


