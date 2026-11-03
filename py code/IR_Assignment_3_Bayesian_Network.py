"""Python code exported from IR_Assignment_3_Bayesian_Network.ipynb."""

# %% Cell 1
# ============================================================
# BAYESIAN NETWORK FOR HEART DISEASE DIAGNOSIS
# Cleveland Heart Disease Dataset
# ============================================================

import pandas as pd
import numpy as np
from google.colab import files
from itertools import product

print("=" * 70)
print("BAYESIAN NETWORK - HEART DISEASE DIAGNOSIS")
print("=" * 70)


# ============================================================
# 1. UPLOAD CLEVELAND HEART DISEASE DATASET
# ============================================================

print("\nUpload your Cleveland Heart Disease CSV file.")

uploaded = files.upload()

filename = list(uploaded.keys())[0]

print("\nUploaded file:", filename)


# ============================================================
# 2. LOAD DATASET
# ============================================================

columns = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'num'
]

data = pd.read_csv(
    filename,
    header=None,
    names=columns
)

print("\nInitial dataset shape:")
print(data.shape)

print("\nFirst 5 records:")
print(data.head())

# %% Cell 2
# ============================================================
# 3. CLEAN DATA
# ============================================================

# Convert '?' to NaN
data = data.replace('?', np.nan)

# Convert all columns to numeric
for column in columns:
    data[column] = pd.to_numeric(
        data[column],
        errors='coerce'
    )

print("\nMissing values:")
print(data.isnull().sum())

# Remove rows containing missing values
data = data.dropna().reset_index(drop=True)

print("\nDataset shape after removing missing values:")
print(data.shape)

# %% Cell 3
# ============================================================
# 4. CONVERT TARGET VARIABLE
# ============================================================

# Cleveland dataset:
# num = 0  -> No heart disease
# num = 1-4 -> Heart disease

data['HeartDisease'] = np.where(
    data['num'] == 0,
    0,
    1
)

data.drop(
    columns=['num'],
    inplace=True
)


# ============================================================
# 5. DISCRETIZE CONTINUOUS VARIABLES
# ============================================================

# Age
data['age'] = pd.cut(
    data['age'],
    bins=[-np.inf, 40, 60, np.inf],
    labels=[0, 1, 2]
).astype(int)

# Resting blood pressure
data['trestbps'] = pd.cut(
    data['trestbps'],
    bins=[-np.inf, 120, 140, np.inf],
    labels=[0, 1, 2]
).astype(int)

# Cholesterol
data['chol'] = pd.cut(
    data['chol'],
    bins=[-np.inf, 200, np.inf],
    labels=[0, 1]
).astype(int)

# Maximum heart rate
data['thalach'] = pd.cut(
    data['thalach'],
    bins=[-np.inf, 120, np.inf],
    labels=[0, 1]
).astype(int)

# ST depression
data['oldpeak'] = pd.cut(
    data['oldpeak'],
    bins=[-np.inf, 1, np.inf],
    labels=[0, 1]
).astype(int)


print("\n" + "=" * 70)
print("CLEANED AND DISCRETIZED DATA")
print("=" * 70)

print(data.head())

print("\nFinal shape:")
print(data.shape)

# %% Cell 4
# ============================================================
# 6. BAYESIAN NETWORK STRUCTURE
# ============================================================

parents = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]

target = 'HeartDisease'


print("\n" + "=" * 70)
print("BAYESIAN NETWORK STRUCTURE")
print("=" * 70)

print("\nEdges:")

for parent in parents:
    print(
        f"{parent} --> {target}"
    )


# ============================================================
# 7. GET STATES
# ============================================================

states = {}

for column in data.columns:
    states[column] = sorted(
        data[column].unique().tolist()
    )

print("\nVariable states:")

for column in states:
    print(
        f"{column}: {states[column]}"
    )


# ============================================================
# 8. CALCULATE PRIOR PROBABILITIES
# ============================================================

target_counts = data[target].value_counts()

total_records = len(data)

prior_probability = {}

for state in states[target]:

    prior_probability[state] = (
        target_counts.get(state, 0)
        / total_records
    )


print("\n" + "=" * 70)
print("PRIOR PROBABILITIES")
print("=" * 70)

for state, probability in prior_probability.items():

    if state == 0:
        label = "No Heart Disease"
    else:
        label = "Heart Disease"

    print(
        f"{label}: {probability:.4f}"
    )


# ============================================================
# 9. BAYESIAN NETWORK CONDITIONAL PROBABILITY
# ============================================================

def calculate_probability(
    patient,
    target_state
):

    # Start with prior probability
    probability = prior_probability[
        target_state
    ]

    target_data = data[
        data[target] == target_state
    ]

    # Naive Bayesian conditional probability
    # P(H | X1,...,Xn) proportional to
    # P(H) * P(X1|H) * ... * P(Xn|H)

    for feature in parents:

        feature_value = patient[feature]

        count_feature_and_target = len(
            target_data[
                target_data[feature]
                == feature_value
            ]
        )

        count_target = len(target_data)

        # Laplace smoothing
        number_of_states = len(
            states[feature]
        )

        conditional_probability = (
            count_feature_and_target + 1
        ) / (
            count_target
            + number_of_states
        )

        probability *= conditional_probability

    return probability


# ============================================================
# 10. NORMALIZE PROBABILITIES
# ============================================================

def diagnose_patient(patient):

    probability_no_disease = (
        calculate_probability(
            patient,
            0
        )
    )

    probability_disease = (
        calculate_probability(
            patient,
            1
        )
    )

    total = (
        probability_no_disease
        + probability_disease
    )

    probability_no_disease /= total
    probability_disease /= total

    if probability_disease > probability_no_disease:

        prediction = "HEART DISEASE"

    else:

        prediction = "NO HEART DISEASE"

    return (
        probability_no_disease,
        probability_disease,
        prediction
    )

# %% Cell 5
# ============================================================
# 11. PATIENT 1
# ============================================================

patient_1 = {
    'age': 2,
    'sex': 1,
    'cp': 4,
    'trestbps': 2,
    'chol': 1,
    'fbs': 0,
    'restecg': 2,
    'thalach': 0,
    'exang': 1,
    'oldpeak': 1,
    'slope': 2,
    'ca': 1,
    'thal': 3
}


print("\n" + "=" * 70)
print("PATIENT 1 DIAGNOSIS")
print("=" * 70)

print("\nPatient data:")

for feature, value in patient_1.items():
    print(
        f"{feature}: {value}"
    )


p_no_1, p_yes_1, prediction_1 = (
    diagnose_patient(patient_1)
)

print("\nProbability of NO heart disease:")
print(
    f"{p_no_1:.4f}"
)

print("\nProbability of HEART disease:")
print(
    f"{p_yes_1:.4f}"
)

print("\nPrediction:")
print(prediction_1)


# ============================================================
# 12. PATIENT 2
# ============================================================

patient_2 = {
    'age': 1,
    'sex': 0,
    'cp': 2,
    'trestbps': 1,
    'chol': 0,
    'fbs': 0,
    'restecg': 0,
    'thalach': 1,
    'exang': 0,
    'oldpeak': 0,
    'slope': 1,
    'ca': 0,
    'thal': 3
}


print("\n" + "=" * 70)
print("PATIENT 2 DIAGNOSIS")
print("=" * 70)

print("\nPatient data:")

for feature, value in patient_2.items():
    print(
        f"{feature}: {value}"
    )


p_no_2, p_yes_2, prediction_2 = (
    diagnose_patient(patient_2)
)

print("\nProbability of NO heart disease:")
print(
    f"{p_no_2:.4f}"
)

print("\nProbability of HEART disease:")
print(
    f"{p_yes_2:.4f}"
)

print("\nPrediction:")
print(prediction_2)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSIS SUMMARY")
print("=" * 70)

print(
    f"\nPatient 1: {prediction_1}"
)

print(
    f"Patient 1 - P(No Disease) = {p_no_1:.4f}"
)

print(
    f"Patient 1 - P(Disease) = {p_yes_1:.4f}"
)

print(
    f"\nPatient 2: {prediction_2}"
)

print(
    f"Patient 2 - P(No Disease) = {p_no_2:.4f}"
)

print(
    f"Patient 2 - P(Disease) = {p_yes_2:.4f}"
)

print("\n" + "=" * 70)
print("BAYESIAN NETWORK DIAGNOSIS COMPLETED")
print("=" * 70)

print(
    "\nNote: This is an educational machine-learning"
    " demonstration and not a clinical diagnosis."
)
