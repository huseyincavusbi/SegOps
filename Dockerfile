# Use official Python image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.serve_clusters:app", "--host", "0.0.0.0", "--port", "8000"]
