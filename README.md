# Loan Prediction Data Science Project

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline for loan prediction analysis. The project follows industry best practices in MLOps, including version control, automated testing, containerization, and deployment capabilities.

### Objective
The primary objective is to build a robust machine learning model that can predict loan approval outcomes based on various applicant features. The project demonstrates end-to-end ML workflow from data preprocessing to model deployment.

### Methodology
- **Data Analysis**: Exploratory data analysis to understand patterns and relationships
- **Feature Engineering**: Creation of relevant features to improve model performance
- **Model Development**: Training and evaluation of multiple ML algorithms
- **Hyperparameter Tuning**: Optimization of model parameters for best performance
- **Model Deployment**: REST API implementation using Flask
- **MLOps Integration**: CI/CD pipeline for continuous model monitoring and deployment

---

## 📂 Project Structure

```
loan/
├── data/
│   ├── raw/                    # Original, immutable data
│   │   └── loan_data.csv
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                        # Source code for the project
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/               # Feature engineering scripts
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── predict.py
│   ├── visualization/          # Visualization scripts
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── models/                     # Trained model artifacts
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── flask_app.py               # Flask API for model serving
├── mlops_pipeline.py          # MLOps CI/CD pipeline
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

---

## 📊 Dataset Overview

### Source
The dataset `loan_data.csv` contains historical loan application data with various applicant features and loan approval outcomes.

### Features
The dataset includes the following types of features:
- **Demographic Information**: Age, gender, marital status, number of dependents
- **Financial Information**: Applicant income, co-applicant income, loan amount, loan term
- **Credit History**: Credit history record
- **Property Information**: Property area (urban, semi-urban, rural)
- **Target Variable**: Loan approval status (approved/rejected)

### Preprocessing Steps
1. **Missing Value Treatment**: Imputation strategies for numerical and categorical variables
2. **Outlier Detection**: Identification and handling of outliers
3. **Feature Encoding**: Label encoding and one-hot encoding for categorical variables
4. **Feature Scaling**: Standardization/normalization of numerical features
5. **Data Splitting**: Train-validation-test split with stratification

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd loan
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Data Exploration
Run the Jupyter notebooks in the `notebooks/` directory:
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 2. Train the Model
```bash
python src/models/train_model.py
```

### 3. Evaluate the Model
```bash
python src/models/evaluate_model.py
```

### 4. Run Flask Application Locally
```bash
python flask_app.py
```
The API will be available at `http://localhost:5000`

### 5. Make Predictions
Send a POST request to the API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
  }'
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t loan-prediction-app .
```

### Run Docker Container
```bash
docker run -p 5000:5000 loan-prediction-app
```

The application will be available at `http://localhost:5000`

---

## 🔄 MLOps Pipeline

### Continuous Integration
The `mlops_pipeline.py` script implements:
- **Automated Testing**: Runs unit tests on code changes
- **Model Validation**: Validates model performance metrics
- **Version Control**: Tracks model versions and experiments
- **Data Validation**: Ensures data quality and schema compliance

### Model Monitoring
- **Performance Tracking**: Monitor model accuracy, precision, recall, F1-score
- **Data Drift Detection**: Detect changes in input data distribution
- **Model Retraining**: Trigger retraining when performance degrades

### Running the MLOps Pipeline
```bash
python mlops_pipeline.py
```

---

## 🧪 Testing

Run all unit tests:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

---

## 📈 Model Development

### Model Selection
The following algorithms were evaluated:
1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble method with good interpretability
3. **XGBoost**: Gradient boosting for superior performance
4. **Support Vector Machine**: For non-linear decision boundaries

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

### Hyperparameter Tuning
- **Grid Search CV**: Exhaustive search over parameter grid
- **Random Search CV**: Random sampling of parameter space
- **Bayesian Optimization**: Intelligent parameter search

---

## 🔧 Configuration

Configuration parameters can be modified in `src/utils/config.py`:
- Model hyperparameters
- Data preprocessing settings
- API configuration
- Logging settings

---

## 📝 API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns the health status of the API.

#### Predict
```
POST /predict
```
**Request Body:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```

**Response:**
```json
{
  "prediction": "Approved",
  "probability": 0.85,
  "status": "success"
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- Your Name - Initial work

---

## 🙏 Acknowledgments

- Dataset source and contributors
- Open-source libraries and frameworks used
- Community support and feedback

---

## 📞 Contact

For questions or feedback, please contact:
- Email: your.email@example.com
- GitHub: @yourusername

---

## 🗺️ Roadmap

- [ ] Implement A/B testing framework
- [ ] Add model explainability (SHAP values)
- [ ] Create web dashboard for predictions
- [ ] Add support for batch predictions
- [ ] Implement model versioning with MLflow
- [ ] Deploy to cloud platforms (AWS/GCP/Azure)
