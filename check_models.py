import os
import joblib
import pandas as pd

df = pd.read_csv(os.path.join("data", "mushroom_dataset_cleaned.csv"))
X = df.drop(columns=["class"])
y = df["class"]

print("First row:")
print(df.iloc[0].to_dict())
print()

model_files = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "SVC": "svc_pipeline.joblib",
    "KNN": "knn_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
}

for name, filename in model_files.items():
    model = joblib.load(os.path.join("models", filename))
    predictions = model.predict(X)
    accuracy = (predictions == y).mean() * 100
    print(f"{name}: accuracy on whole CSV = {accuracy:.2f}%, first row -> {predictions[0]} (actual: {y.iloc[0]})")