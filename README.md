# 📉 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-ffaa00)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

A professional, end-to-end Machine Learning solution designed to predict customer churn in the telecommunications sector. This system provides a robust ML pipeline, a high-performance RESTful API for real-time inference, and an interactive analytics dashboard.

---

## 🚀 Features

- **End-to-End ML Pipeline**: Automated data ingestion, preprocessing, feature engineering, and model training.
- **Advanced Predictive Modeling**: Utilizes LightGBM and Random Forest algorithms for highly accurate churn classification.
- **RESTful API**: FastAPI-powered inference engine with built-in health checks and OpenAPI documentation.
- **Interactive Dashboard**: Streamlit interface offering comprehensive churn analytics, feature importance visualizations, and real-time prediction capabilities.
- **Production-Ready Containerization**: Fully dockerized multi-service architecture (API + Dashboard) via Docker Compose.

---

## 🏗 System Architecture

The project is structured into distinct, scalable layers:
1. **Data Layer**: Telco Customer Churn data simulating realistic distributions.
2. **ML Pipeline Layer (`src/`)**: Handles data transformation, feature engineering, and robust model training.
3. **Inference Layer (`api/`)**: Evaluates new customer profiles against the trained model via a high-throughput REST API.
4. **Presentation Layer (`dashboard/`)**: Consumes the API to deliver actionable insights and predictions to end-users.

---

## 🛠 Tech Stack

- **Machine Learning**: `scikit-learn`, `lightgbm`, `pandas`, `numpy`
- **Backend API**: `FastAPI`, `uvicorn`, `pydantic`
- **Frontend / Dashboard**: `Streamlit`, `plotly`
- **Deployment**: `Docker`, `Docker Compose`

---

## ⚡ Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) (Recommended)
- Python 3.9+ (Local development)

### Running with Docker (Production Mode)

The easiest way to get the system up and running is via Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

# Build and start the services
docker-compose up --build
```
- **API Documentation (Swagger UI)**: http://localhost:8000/docs
- **Interactive Dashboard**: http://localhost:8501

### Local Setup (Development Mode)

```bash
# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (will generate artifacts in /models)
python src/train_model.py

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# In a new terminal, start the dashboard
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```text
├── api/
│   └── main.py              # FastAPI inference application
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
├── data/
│   └── telco_churn.csv      # Customer dataset
├── models/                  # Serialized model artifacts (joblib)
├── src/
│   ├── __init__.py 
│   ├── data_processing.py   # Data cleaning and preprocessing
│   └── train_model.py       # Model training and evaluation script
├── Dockerfile               # Multi-stage Docker configuration
├── docker-compose.yml       # Service orchestration (API + Dashboard)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Model Evaluation
The ML pipeline seamlessly evaluates core algorithms including Logistic Regression, Random Forest, and LightGBM models. The best performing model is dynamically selected and saved based on the **ROC-AUC** metric, ensuring maximum predictive reliability on highly imbalanced churn data.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ for data-driven customer success.*
