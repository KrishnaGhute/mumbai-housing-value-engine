# 🏙️ Mumbai Housing Value Engine

### Machine Learning · Random Forest Regression · Flask · Geospatial Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Geospatial%20Visualization-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Status](https://img.shields.io/badge/Status-Development-blue)]()

**Mumbai Housing Value Engine** is a machine learning web application designed to estimate housing values using property-related features and geographical information. It combines a trained Random Forest regression model with a Flask-based interface and map-oriented visualization to make model predictions accessible through an interactive application.

The project demonstrates an end-to-end machine learning workflow, from data preparation and model evaluation to application development and deployment preparation.

**Repository:** [KrishnaGhute/mumbai-housing-value-engine](https://github.com/KrishnaGhute/mumbai-housing-value-engine)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Application Screenshots](#-application-screenshots)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Performance](#-model-performance)
- [Feature Importance](#-feature-importance)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Model File Setup](#-model-file-setup)
- [Dataset Information](#-dataset-information)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 📌 Overview

Housing valuation is influenced by several factors, including geographical location, income characteristics, housing age, and the availability of residential infrastructure.

This project explores how supervised machine learning can estimate housing values from structured housing data.

The application uses a trained **Random Forest Regressor** to generate predictions and presents them through a web interface. Geographical information supports map-based exploration of housing locations and model estimates.

### Project Objectives

- Develop a machine learning model for housing value prediction.
- Prepare and analyze structured housing data.
- Compare regression algorithms and evaluate model performance.
- Build a web interface using Flask.
- Present predictions through a user-friendly interface.
- Explore the use of geographical information in housing analysis.

> **Note:** Model estimates are statistical predictions, not official property valuations or guaranteed market prices.

---

## 🚀 Key Features

### 1. Housing Value Prediction
Uses a trained Random Forest regression model to estimate housing values from input features.

### 2. Interactive Web Interface
Provides a browser-based interface for entering property-related information and viewing predictions.

### 3. Geospatial Visualization
Includes map-oriented visualization to help explore geographical patterns in housing values.

### 4. Machine Learning Model Evaluation
Documents regression performance using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

### 5. Reusable Model
Separates the trained model from the application code so that the model can be loaded and reused without retraining for every prediction.

### 6. Modular Application Structure
Organizes application logic, HTML templates, styling, assets, model metrics, and dependencies into separate files and directories.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Model training, tuning, and evaluation |
| Random Forest Regressor | Housing value prediction |
| Flask | Web application backend |
| HTML5 | Web page structure |
| CSS3 | Interface styling |
| Plotly | Interactive visualizations |
| OpenStreetMap | Map visualization and geographical context |
| Joblib / Pickle-compatible serialization | Model persistence, depending on the implementation |
| Git and GitHub | Version control and source code management |

The application dependencies are listed in [`requirements.txt`](requirements.txt).

---

## 🖥️ Application Screenshots

### Home Page

![Mumbai Housing Value Engine - Home Page](assets/screenshots/home.png)

### Geospatial Map

![Mumbai Housing Value Engine - Map View](assets/screenshots/map.png)

### Prediction Results

![Mumbai Housing Value Engine - Prediction Results](assets/screenshots/prediction.png)

---

## 🧠 Machine Learning Workflow

The project follows a supervised machine learning workflow.

1. **Data collection:** Load the housing dataset.
2. **Data preprocessing:** Inspect missing values, data types, and data quality.
3. **Exploratory data analysis:** Examine feature distributions and relationships.
4. **Feature preparation:** Prepare numerical and categorical features for model training.
5. **Train-test split:** Separate data for model development and evaluation.
6. **Model experimentation:** Evaluate regression algorithms, including Random Forest and other candidate models.
7. **Hyperparameter tuning:** Search for a suitable Random Forest configuration.
8. **Model evaluation:** Measure predictive performance on held-out data.
9. **Model serialization:** Save the trained estimator for later inference.
10. **Application integration:** Load the model in Flask and expose predictions through the web interface.

### Selected Random Forest Configuration

The tuned configuration recorded for this project is:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 30 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | 1.0 |

These settings describe the selected model configuration; results can vary if the dataset, preprocessing, or training procedure changes.

---

## 📊 Model Performance

The following results were recorded during model evaluation.

| Metric | Result |
|---|---:|
| Test R² | 0.964864 |
| Test MAE | 235,582.50 |
| Test RMSE | 475,450.73 |
| Cross-validation R² | 0.957565 |

### What Do These Metrics Mean?

- **R² (coefficient of determination):** Measures how well the model explains variation in the target values. A score closer to 1 generally indicates a stronger fit.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual values.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily than MAE.
- **Cross-validation R²:** Helps assess how consistently the model performs across validation folds.

The reported test R² of approximately **0.965** indicates strong predictive performance on the evaluated test split. However, this result alone does not establish real-world accuracy for current Mumbai property transactions.

The monetary interpretation of MAE and RMSE depends on the target variable's units and the dataset's provenance.

---

## 📈 Feature Importance

The recorded Random Forest feature-importance values indicate the following leading features:

| Feature | Importance |
|---|---:|
| `median_income` | 0.728955 |
| `latitude` | 0.131883 |
| `longitude` | 0.116782 |

These values represent the model's relative feature importance, not causal effects.

The results suggest that income-related information and geographical coordinates contributed substantially to the trained model's predictions. Feature importance does not prove that a feature independently causes a housing price to increase or decrease.

---

## 🏗️ Project Architecture

```text
User
  |
  v
Web Browser
  |
  v
Flask Application (app.py)
  |
  +----> Input Validation
  |
  +----> Trained Random Forest Model
  |          |
  |          v
  |      Housing Value Prediction
  |
  +----> HTML Templates
  |
  +----> Static CSS and Visual Assets
  |
  v
Prediction Results / Map Interface
```

The model is stored separately from the source code and must be available locally when the application loads it.

---

## 📂 Project Structure

```text
mumbai-housing-value-engine/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── model_metrics.json
│
├── assets/
│   ├── logo.png
│   └── screenshots/
│       ├── home.png
│       ├── map.png
│       └── prediction.png
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── mumbai_housing_rf_model.pkl   # Local model file; not tracked by Git
```

The trained model file is excluded from the GitHub repository because it exceeds GitHub's regular 100 MB per-file limit.

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version supported by your dependencies
- Git
- A code editor such as Visual Studio Code
- A trained model file compatible with the application

### Step 1: Clone the Repository

```bash
git clone https://github.com/KrishnaGhute/mumbai-housing-value-engine.git
```

Navigate to the project directory:

```bash
cd mumbai-housing-value-engine
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv mumbai
```

Activate it:

```powershell
.\mumbai\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution-policy setting for the current process or activate the environment using another supported method.

### Step 3: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the Trained Model

Obtain the compatible trained model file and place it in the project root:

```text
mumbai-housing-value-engine/
└── mumbai_housing_rf_model.pkl
```

See [Model File Setup](#-model-file-setup) for details.

---

## ▶️ Running the Application

After installing dependencies and placing the model file in the expected location, run:

```powershell
python app.py
```

Open the local URL printed by Flask in your terminal. For a typical Flask development configuration, this may be:

```text
http://127.0.0.1:5000
```

The exact address and port depend on the settings in `app.py`.

Use the interface to enter the available housing features and review the generated prediction.

**Development note:** Flask's development server is intended for local testing, not production hosting.

---

## 📦 Model File Setup

The trained model file, `mumbai_housing_rf_model.pkl`, is approximately 139 MB and is intentionally excluded from this GitHub repository.

To run the application, obtain the exact compatible model artifact from the project owner or generate it using the original training pipeline.

Before loading a model, ensure that it comes from a trusted source. Pickle-based model files can execute code during deserialization, so never load an untrusted artifact.

### Why Is the Model Not Included?

- It exceeds GitHub's regular 100 MB file-size limit.
- Excluding it keeps the source repository smaller.
- The model can be stored separately and supplied when needed.

A future release can use Git LFS or an appropriate model-storage service to distribute large artifacts.

---

## 🗃️ Dataset Information

The model development used a housing dataset associated with the `hindav/Mumbai-Housing-Prices` dataset source.

The dataset contains housing-related attributes such as:

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of housing within an area |
| `total_rooms` | Total rooms recorded for the area |
| `total_bedrooms` | Total bedrooms recorded for the area |
| `population` | Population associated with the area |
| `households` | Number of households |
| `median_income` | Median income feature |
| `ocean_proximity` | Categorical proximity feature |
| `median_house_value` | Target variable for prediction |

The working dataset contained approximately 20,000 records.

**Data interpretation:** These fields describe aggregated geographical areas rather than a complete specification of each individual property. Confirm the dataset's geographical coverage, target-value units, and licensing before interpreting predictions as Mumbai-specific market prices.

---

## ⚠️ Limitations

- Predictions depend on the quality, representativeness, and geographical coverage of the training data.
- Historical housing data may not reflect current market conditions.
- The model does not guarantee an individual property's actual selling price.
- Aggregate area-level features cannot capture every property-specific factor, such as floor, condition, amenities, or building quality.
- Feature importance describes the fitted model and should not be interpreted as causal evidence.
- The serialized model must be compatible with the application's preprocessing and inference logic.
- The reported test metrics apply to the evaluated dataset and split; independent validation is needed before practical use.

---

## 🔮 Future Improvements

- Add explainability using SHAP.
- Improve data validation and error handling.
- Add additional property-level features where reliable data is available.
- Evaluate performance across geographical subregions.
- Add prediction uncertainty or error intervals.
- Automate model training and evaluation.
- Add unit tests and integration tests.
- Configure production-grade deployment.
- Store large model artifacts using Git LFS or external model storage.
- Document reproducible training, preprocessing, and evaluation procedures.

---

## 👨‍💻 Author

**Krishna Ghute**

Data Science professional interested in machine learning, deep learning, geospatial analytics, and practical data-driven applications.

- **GitHub:** [KrishnaGhute](https://github.com/KrishnaGhute)
- **LinkedIn:** [Krishna Ghute](https://www.linkedin.com/in/krishna-ghute-b72199370/)
- **Kaggle:** [krishnaghuteds](https://www.kaggle.com/krishnaghuteds)
- **YouTube:** [@Krishna_Ghute](https://www.youtube.com/@Krishna_Ghute)

---

## 📄 License

No license has been specified for this repository yet. Unless a license is added, reuse and redistribution are subject to applicable copyright law.

---

**Built with Python, machine learning, and a focus on practical implementation.**
