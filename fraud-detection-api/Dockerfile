# Using an official lightweight Python image
FROM python:3.11-slim

# Setting the working directory in the container
WORKDIR /app

# Copying the local project files to the container's working directory
COPY . .

# Installation of the Python dependencies
RUN pip install "fastapi[all]" mlflow "scikit-learn==1.5.1" "xgboost==2.1.0" "pandas==2.2.2" "imbalanced-learn==0.12.3"

# Command to run the application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]