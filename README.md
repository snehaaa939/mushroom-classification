
# Mushroom Classification 🍄

## Project Overview
This machine learning project classifies mushrooms as **edible or poisonous** using their physical characteristics. It uses supervised learning to compare multiple classification algorithms and evaluate their performance.

## Dataset
- **Source:** Mushroom dataset
- **Total records:** 8,124
- **Input features:** 21 categorical features
- **Target:** `class` — edible (`e`) or poisonous (`p`)
- **Preprocessing:** Replaced unknown stalk-root values with an `unknown` category and removed a constant feature.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the class distribution and relationships between mushroom characteristics and their classes. A profiling report was also generated.

## Machine Learning Workflow
1. Data loading and cleaning
2. Exploratory Data Analysis and visualization
3. Train-test split
4. Categorical feature preprocessing using `OneHotEncoder` and `ColumnTransformer`
5. Model pipelines using scikit-learn
6. Hyperparameter tuning with `GridSearchCV`
7. Model evaluation and comparison
8. Saving trained pipelines with Joblib

## Models Evaluated
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

## Results
The models achieved test accuracies of approximately **99.94% to 100%** on the selected test split. These results apply to this dataset and evaluation setup; they do not guarantee perfect predictions on new, real-world mushrooms.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- YData Profiling
- Joblib
- Jupyter Notebook

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
├── mushroom.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run
1. Clone or download this repository.
2. Install the required packages listed in `requirements.txt`.
3. Open `mushroom.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells in order.

## Disclaimer
This project is for educational purposes only. Never use a machine learning prediction as the sole basis for deciding whether a wild mushroom is safe to eat.
