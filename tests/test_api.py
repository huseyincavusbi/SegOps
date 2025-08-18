import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.serve_clusters import app

def test_predict_cluster():
    payload = {
        "Booking_Value": 100,
        "Ride_Distance": 10,
        "Driver_Ratings": 4.5,
        "Customer_Rating": 4.0,
        "Vehicle_Type": "Sedan",
        "Payment_Method": "Credit Card"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "cluster" in data
        assert isinstance(data["cluster"], int)
