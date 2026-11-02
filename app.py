#! Step 1: Imports and page configuration

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#! Step 2: Project paths and load data/models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mushroom_dataset_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSS_PATH = os.path.join(BASE_DIR, "style.css")

df = pd.read_csv(DATA_PATH)
feature_columns = df.drop(columns=["class"]).columns.tolist()

model_files = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "SVC": "svc_pipeline.joblib",
    "KNN": "knn_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
}

models = {
    name: joblib.load(os.path.join(MODEL_DIR, filename))
    for name, filename in model_files.items()
}

# Load styling

if os.path.exists(CSS_PATH):
    with open(CSS_PATH, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )

#!Step 3: Feature labels

feature_labels = {
    "cap-shape": "Cap Shape",
    "cap-surface": "Cap Surface",
    "cap-color": "Cap Color",
    "bruises": "Bruises",
    "odor": "Odor",
    "gill-attachment": "Gill Attachment",
    "gill-spacing": "Gill Spacing",
    "gill-size": "Gill Size",
    "gill-color": "Gill Color",
    "stalk-shape": "Stalk Shape",
    "stalk-root": "Stalk Root",
    "stalk-surface-above-ring": "Stalk Surface Above Ring",
    "stalk-surface-below-ring": "Stalk Surface Below Ring",
    "stalk-color-above-ring": "Stalk Color Above Ring",
    "stalk-color-below-ring": "Stalk Color Below Ring",
    "veil-color": "Veil Color",
    "ring-number": "Number of Rings",
    "ring-type": "Ring Type",
    "spore-print-color": "Spore Print Color",
    "population": "Population",
    "habitat": "Habitat",
}


#! Step 4: Human-readable category descriptions
# Original category codes are passed to the trained models.

feature_options = {
    "cap-shape": {
        "b": "Bell",
        "c": "Conical",
        "x": "Convex",
        "f": "Flat",
        "k": "Knobbed",
        "s": "Sunken",
    },
    "cap-surface": {
        "f": "Fibrous",
        "g": "Grooves",
        "y": "Scaly",
        "s": "Smooth",
    },
    "cap-color": {
        "n": "Brown",
        "y": "Yellow",
        "w": "White",
        "g": "Gray",
        "e": "Red",
        "p": "Pink",
        "b": "Buff",
        "u": "Purple",
        "c": "Cinnamon",
        "r": "Green",
    },
    "bruises": {"t": "Yes", "f": "No"},
    "odor": {
        "p": "Pungent",
        "a": "Almond",
        "l": "Anise",
        "n": "None",
        "f": "Foul",
        "c": "Creosote",
        "y": "Fishy",
        "s": "Spicy",
        "m": "Musty",
    },
    "gill-attachment": {
        "a": "Attached",
        "d": "Descending",
        "f": "Free",
        "n": "Notched",
    },
    "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {
        "k": "Black",
        "n": "Brown",
        "b": "Buff",
        "h": "Chocolate",
        "g": "Gray",
        "r": "Green",
        "o": "Orange",
        "p": "Pink",
        "u": "Purple",
        "e": "Red",
        "w": "White",
        "y": "Yellow",
    },
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {
        "b": "Bulbous",
        "c": "Club",
        "u": "Cup",
        "e": "Equal",
        "z": "Rhizomorphs",
        "r": "Rooted",
        "unknown": "Unknown",
    },
    "stalk-surface-above-ring": {
        "f": "Fibrous",
        "y": "Scaly",
        "k": "Silky",
        "s": "Smooth",
    },
    "stalk-surface-below-ring": {
        "f": "Fibrous",
        "y": "Scaly",
        "k": "Silky",
        "s": "Smooth",
    },
    "stalk-color-above-ring": {
        "n": "Brown",
        "b": "Buff",
        "c": "Cinnamon",
        "g": "Gray",
        "o": "Orange",
        "p": "Pink",
        "e": "Red",
        "w": "White",
        "y": "Yellow",
    },
    "stalk-color-below-ring": {
        "n": "Brown",
        "b": "Buff",
        "c": "Cinnamon",
        "g": "Gray",
        "o": "Orange",
        "p": "Pink",
        "e": "Red",
        "w": "White",
        "y": "Yellow",
    },
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {
        "c": "Cobwebby",
        "e": "Evanescent",
        "f": "Flaring",
        "l": "Large",
        "n": "None",
        "p": "Pendant",
        "s": "Sheathing",
        "z": "Zone",
    },
    "spore-print-color": {
        "k": "Black",
        "n": "Brown",
        "b": "Buff",
        "h": "Chocolate",
        "r": "Green",
        "o": "Orange",
        "u": "Purple",
        "w": "White",
        "y": "Yellow",
    },
    "population": {
        "a": "Abundant",
        "c": "Clustered",
        "n": "Numerous",
        "s": "Scattered",
        "v": "Several",
        "y": "Solitary",
    },
    "habitat": {
        "g": "Grasses",
        "l": "Leaves",
        "m": "Meadows",
        "p": "Paths",
        "u": "Urban",
        "w": "Waste",
        "d": "Woods",
    },
}


#! Step 5: Group related input features

feature_groups = {
    "🍄 Cap Characteristics": [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
    ],
    "🌸 Gill Characteristics": [
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
    ],
    "🌿 Stalk Characteristics": [
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
    ],
    "🔎 Other Characteristics": [
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ],
}


#! Step 6: App header and overview

st.markdown(
    """
    <div class="hero">
        <h1>🍄 Mushroom Classification</h1>
        <p>Explore mushroom characteristics and use four trained machine
        learning models to predict whether a mushroom is edible or poisonous.</p>
        <span class="badge">4 ML Models</span>
        <span class="badge">21 Features</span>
        <span class="badge">Educational Project</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

st.subheader("📊 Project Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Dataset Samples", f"{len(df):,}")
col2.metric("Input Features", len(feature_columns))
col3.metric("Trained Models", len(models))
st.divider()


#! Step 7: Data description

st.header("📁 Dataset Description")
st.write(
    "The mushroom dataset contains categorical descriptions of mushroom "
    "physical characteristics, including cap, gill, stalk, ring, spore-print, "
    "population, and habitat attributes. The target column is `class`."
)
st.write(
    "**Prediction target:** `e` represents edible and `p` represents poisonous. "
    "The models use the 21 input characteristics to predict this target class."
)
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True, hide_index=True)

#! Step 8: EDA section

st.header("🔍 Exploratory Data Analysis")
st.write(
    "Explore the distribution of the target classes and the values in "
    "individual mushroom features."
)

st.subheader("Target Class Distribution")

target_df = pd.DataFrame(
    {
        "Edible": [(df["class"] == "e").sum()],
        "Poisonous": [(df["class"] == "p").sum()],
    }
)
target_df.index = [" "]
st.bar_chart(target_df, color=["#1A820AFF", "#DF0113E6"], stack=False)

eda_feature = st.selectbox(
    "Choose a feature to explore",
    feature_columns,
    format_func=lambda col: feature_labels.get(col, col.replace("-", " ").title()),
    key="eda_feature",
)

options_map = feature_options.get(eda_feature, {})

feature_distribution = df[eda_feature].fillna("Missing").astype(str).value_counts()

feature_distribution.index = [
    options_map.get(value, value) for value in feature_distribution.index
]

st.subheader(f"{feature_labels.get(eda_feature, eda_feature)} Distribution")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(feature_distribution.index, feature_distribution.values, color="#315C28")
ax.tick_params(axis="x", labelrotation=0)
ax.set_xlabel("Category")
ax.set_ylabel("Count")
fig.patch.set_facecolor("#F6F1E7")  # page color around the chart
ax.set_facecolor("#F6F1E7")  # chart area color
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# YData Profiling Report

st.subheader("YData Profiling Report")

report_path = os.path.join(BASE_DIR, "data", "mushroom_cleaned.html")

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as file:
        report_html = file.read()

    st.download_button(
        "Download EDA Report (open in browser)",
        data=report_html,
        file_name="mushroom_eda_report.html",
        mime="text/html",
    )
else:
    st.warning("YData Profiling report not found.")

#! Step 9: Mushroom identification form

st.header("🔎 Predict Mushroom Class")
st.write(
    "Choose one value for each characteristic. The displayed descriptions "
    "include the original dataset codes used by the models."
)

if "form_version" not in st.session_state:
    st.session_state["form_version"] = 0

with st.form("prediction_form"):
    input_data = {}

    for group_name, group_features in feature_groups.items():
        available_features = [
            column for column in group_features if column in feature_columns
        ]
        if not available_features:
            continue

        st.markdown(f"### {group_name}")
        left, right = st.columns(2)

        for i, column in enumerate(available_features):
            categories = sorted(
                df[column].dropna().unique().tolist(),
                key=str,
            )
            label = feature_labels.get(column, column.replace("-", " ").title())
            options_map = feature_options.get(column, {})

            def format_option(value, mapping=options_map):
                description = mapping.get(value)
                return f"{description} ({value})" if description else f"{value} (code)"

            with left if i % 2 == 0 else right:
                input_data[column] = st.selectbox(
                    label,
                    options=categories,
                    format_func=format_option,
                    key=f"input_{column}_{st.session_state['form_version']}",
                )

    submitted = st.form_submit_button(
        "🔍 Predict with All Models",
        type="primary",
        use_container_width=True,
    )


#! Step 10: Show predictions from all four models

if submitted:
    input_df = pd.DataFrame(
        [[input_data[column] for column in feature_columns]],
        columns=feature_columns,
    )

    st.divider()
    st.subheader("🤖 Predictions from All Models")

    result_columns = st.columns(len(models))
    predictions = {}

    for (model_name, model), result_column in zip(models.items(), result_columns):
        prediction_code = str(model.predict(input_df)[0]).lower()
        prediction_label = {
            "e": "Edible",
            "p": "Poisonous",
        }.get(prediction_code, prediction_code)

        predictions[model_name] = prediction_label

        with result_column:
            st.markdown(f"**{model_name}**")
            if prediction_code == "e":
                st.success("🍃 Edible")
            elif prediction_code == "p":
                st.error("⚠️ Poisonous")
            else:
                st.info(prediction_label)

    if len(set(predictions.values())) == 1:
        agreed_prediction = next(iter(predictions.values()))
        st.info(f"All four models predict: **{agreed_prediction}**")
    else:
        st.warning(
            "The models do not all agree. Review each model's prediction "
            "rather than relying on a single result."
        )

    st.warning(
        "Safety notice: This application is for educational purposes only. "
        "Machine learning predictions are not a reliable way to identify "
        "wild mushrooms. Never eat a mushroom based on this app."
    )

    def reset_mushroom_form():
        st.session_state["form_version"] += 1

    st.button(
        "Try Another Mushroom",
        type="primary",
        on_click=reset_mushroom_form,
    )

#! Step 11: Model accuracy comparison

st.header("📊 Model Accuracy Comparison")
st.write(
    "The following test accuracies are the recorded evaluation results "
    "from the model training stage."
)

model_comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "SVC",
            "KNN",
            "Random Forest",
        ],
        "Test Accuracy (%)": [99.94, 100.00, 100.00, 100.00],
    }
)

st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True,
)
st.bar_chart(model_comparison, x="Model", y="Test Accuracy (%)", color="#8B5E47FF")
st.caption(
    "All four models score between 99.94% and 100%, so the bars look the "
    "same. The table above shows the exact values."
)
