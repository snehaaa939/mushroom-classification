# Mushroom Classification 🍄

## Project Overview

This machine learning project classifies mushrooms as **edible or poisonous** based on their physical characteristics.

It uses supervised machine learning to train and compare multiple classification algorithms and presents their predictions through an interactive Streamlit dashboard.

## Dataset

- **Source:** Mushroom Dataset (add the exact Kaggle or UCI dataset link)
- **Total records:** 8,124
- **Input features:** 21 categorical features
- **Target variable:** `class`
  - `e` — Edible
  - `p` — Poisonous
- **Preprocessing:**
  - Replaced unknown values in `stalk-root` with an `unknown` category.
  - Removed a constant feature (`veil-type`).

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset, examine the target class distribution, and explore relationships between mushroom characteristics and their classes.

A data profiling report was also generated using **YData Profiling**.

## Machine Learning Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis and visualization
4. Train-test split
5. Categorical feature encoding and preprocessing
6. Model training using scikit-learn pipelines
7. Hyperparameter tuning with `GridSearchCV`
8. Model evaluation and comparison
9. Saving trained pipelines using Joblib
10. Integrating the trained models into a Streamlit dashboard

## Models Evaluated

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

## Results

The models achieved test accuracies of approximately **99.94% to 100%** on the selected test split.

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 99.94% |
| Support Vector Classifier | 100% |
| K-Nearest Neighbors | 100% |
| Random Forest | 100% |

These results reflect performance on this dataset and evaluation setup. They do not guarantee the same performance on unseen, real-world mushrooms.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides:

- Dataset description and preview
- Exploratory data analysis and visualizations
- Model accuracy comparison
- Mushroom feature selection through input controls
- Predictions from all four trained models

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- YData Profiling
- Joblib
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
Mushroom_classification/
├── data/
│   └── mushroom_dataset_cleaned.csv
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── svc_pipeline.joblib
│   ├── knn_pipeline.joblib
│   └── random_forest_pipeline.joblib
├── app.py
├── style.css
├── mushroom.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/snehaaa939/mushroom-classification.git
```

### 2. Navigate to the project folder

```bash
cd mushroom-classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

You can also open `mushroom.ipynb` in Jupyter Notebook or VS Code to explore the data analysis and model training process.

## Disclaimer

This project is developed for educational purposes only. Machine learning predictions should never be used as the sole basis for determining whether a wild mushroom is safe to eat.