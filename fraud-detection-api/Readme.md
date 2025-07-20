# End-to-End Fraud Detection API: An MLOps Project

This repository documents a complete, end-to-end MLOps project to build, deploy, and monitor a real-time fraud detection API. It demonstrates the entire lifecycle, from model training to a live, auto-scaling cloud deployment.

---

## 🏗️ System Architecture

The project follows a modern MLOps workflow, integrating key technologies to automate the path from code to a live, scalable service.

Local Development ➡️ Git Push ➡️ GitHub Actions (CI/CD) ➡️ Docker Hub (Registry) ➡️ Google Cloud Run (Deployment) ➡️ Live API

---

## 🚀 Project Phases

We built this project in five distinct phases, covering the entire machine learning lifecycle.

### ### Phase 1: Model Development & Training 🧠

* **Goal**: To train a robust machine learning model capable of detecting fraudulent credit card transactions.
* **Process**:
    1.  **Data Loading & Preprocessing**: The project uses the standard `creditcard.csv` dataset. The `Time` and `Amount` features were standardized using `StandardScaler` to prepare them for modeling.
    2.  **Handling Imbalance**: The dataset is highly imbalanced. We addressed this by applying **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data, creating a balanced set for the model to learn from without biasing the evaluation.
    3.  **Model Training**: An **XGBoost Classifier** was chosen for its high performance and speed on tabular data.
    4.  **Artifact Generation**: The final trained model was saved as a single `model.joblib` file using the `joblib` library, which is efficient for scikit-learn compatible models.
* **Key File**: `fraud-detection-api/train.py`

---

### ### Phase 2: API Development & Containerization 📦

* **Goal**: To wrap our trained model in a scalable, portable API service.
* **Process**:
    1.  **API Creation**: A web server was built using **FastAPI** for its high performance and automatic documentation generation. It exposes a `/predict` endpoint that accepts transaction data in JSON format and returns a fraud prediction.
    2.  **Model Loading**: On startup, the FastAPI application loads the `model.joblib` file directly from its local directory.
    3.  **Containerization**: A **`Dockerfile`** was written to create a self-contained Docker image. This image includes the Python environment, all necessary libraries, and our application code, ensuring it runs consistently anywhere.
* **Key Files**: `fraud-detection-api/main.py`, `fraud-detection-api/Dockerfile`

---

### ### Phase 3: CI/CD Automation with GitHub Actions ⚙️

* **Goal**: To automate the process of building and publishing our Docker image whenever new code is pushed to the repository.
* **Process**:
    1.  **Workflow Definition**: A GitHub Actions workflow was created at `.github/workflows/fraud-detection-cicd.yml`.
    2.  **Trigger**: The workflow is configured to run automatically only when changes are made inside the `fraud-detection-api/` directory of the `main` branch.
    3.  **Pipeline Steps**:
        * The job checks out the latest code from the repository.
        * It securely logs into **Docker Hub** using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` stored as GitHub Secrets.
        * It builds the Docker image using our `Dockerfile`.
        * It pushes the finished image to a public Docker Hub repository, tagging it as `:latest`.
* **Key Concept**: This pipeline creates a production-ready artifact automatically, a core principle of Continuous Integration and Continuous Deployment.

---

### ### Phase 4: Cloud Deployment ☁️

* **Goal**: To deploy our containerized API to the cloud and make it publicly accessible.
* **Process**:
    1.  **Platform Choice**: We used **Google Cloud Run**, a serverless platform that automatically manages infrastructure and scaling, making it cost-effective and easy to manage.
    2.  **Deployment**: A new Cloud Run service was created directly from the Docker image we published to **Docker Hub**.
    3.  **Configuration**: The service was configured to be publicly accessible by **allowing unauthenticated invocations** and was set to listen on the correct internal container **port (8000)**.
* **Result**: A live, scalable, secure HTTPS endpoint that serves our fraud detection model to the world.

---

### ### Phase 5: Monitoring Strategy 📈

* **Goal**: To understand how to monitor the health and performance of our deployed service.
* **Process (Conceptual)**:
    1.  **Operational Monitoring**: We identified that key service metrics like **request latency, request count, and server errors** are automatically available in the **"Metrics" tab** of our Google Cloud Run service dashboard.
    2.  **Model Performance Monitoring**: We outlined a strategy to detect **Model Drift**. This involves modifying the API to log all incoming predictions to a service like Google Cloud Logging. A separate, scheduled job would then analyze these logs to detect statistical changes in the data over time, which would signal the need to retrain the model.

---

## 🛠️ How to Run Locally

1.  Clone this repository.
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  Navigate to the project folder.
    ```bash
    cd YOUR_REPO_NAME/fraud-detection-api
    ```
3.  Set up your Kaggle API key (`kaggle.json`) to allow data download.
4.  Run the training script to generate the `model.joblib` file.
    ```bash
    python train.py
    ```
5.  Build the Docker image.
    ```bash
    docker build -t fraud-api .
    ```
6.  Run the container.
    ```bash
    docker run -p 8000:8000 fraud-api
    ```
7.  Access the interactive API documentation in your browser at `http://localhost:8000/docs`.
