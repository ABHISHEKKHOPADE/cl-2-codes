"""Python code exported from BI_Assignment_2_RNASeq_Data_Analysis.ipynb."""

# %% Cell 1
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Create RNA-Seq Dataset Directly in the Program
# ------------------------------------------------------------

data_dict = {
    "Gene": [
        "Gene1", "Gene2", "Gene3", "Gene4", "Gene5",
        "Gene6", "Gene7", "Gene8", "Gene9", "Gene10"
    ],

    "Control_1": [
        100, 500, 50, 800, 200,
        1000, 75, 450, 300, 150
    ],

    "Control_2": [
        110, 480, 55, 820, 210,
        980, 80, 460, 310, 160
    ],

    "Control_3": [
        105, 510, 52, 790, 205,
        1020, 78, 440, 295, 155
    ],

    "Treatment_1": [
        250, 200, 60, 400, 600,
        990, 300, 100, 700, 500
    ],

    "Treatment_2": [
        260, 210, 58, 420, 620,
        1010, 310, 110, 720, 510
    ],

    "Treatment_3": [
        245, 190, 62, 410, 610,
        1005, 295, 105, 680, 490
    ]
}

data = pd.DataFrame(data_dict)

# %% Cell 2
# ------------------------------------------------------------
# 2. Display Dataset
# ------------------------------------------------------------

print("=" * 65)
print("                 RNA-SEQ DATA ANALYSIS")
print("=" * 65)

print("\nRNA-Seq Dataset:")
print(data.to_string(index=False))


# ------------------------------------------------------------
# 3. Basic Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                  DATASET INFORMATION")
print("=" * 65)

print("Number of genes:", len(data))
print("Number of samples:", len(data.columns) - 1)

print("\nMissing values:")
print(data.isnull().sum())


# ------------------------------------------------------------
# 4. Define Control and Treatment Groups
# ------------------------------------------------------------

control_samples = [
    "Control_1",
    "Control_2",
    "Control_3"
]

treatment_samples = [
    "Treatment_1",
    "Treatment_2",
    "Treatment_3"
]

# %% Cell 3
# ------------------------------------------------------------
# 5. Calculate Mean Expression
# ------------------------------------------------------------

data["Control_Mean"] = data[control_samples].mean(axis=1)

data["Treatment_Mean"] = data[treatment_samples].mean(axis=1)


# ------------------------------------------------------------
# 6. Calculate Log2 Fold Change
# ------------------------------------------------------------

# Pseudocount prevents division by zero

pseudo_count = 1

data["Log2_Fold_Change"] = np.log2(
    (data["Treatment_Mean"] + pseudo_count) /
    (data["Control_Mean"] + pseudo_count)
)


# ------------------------------------------------------------
# 7. Perform Statistical Test
# ------------------------------------------------------------

p_values = []

for index, row in data.iterrows():

    control_values = row[control_samples].astype(float)

    treatment_values = row[treatment_samples].astype(float)

    statistic, p_value = ttest_ind(
        treatment_values,
        control_values,
        equal_var=False
    )

    p_values.append(p_value)

data["P_Value"] = p_values


# ------------------------------------------------------------
# 8. Multiple Testing Correction
# ------------------------------------------------------------

data["Adjusted_P_Value"] = multipletests(
    data["P_Value"],
    method="fdr_bh"
)[1]


# ------------------------------------------------------------
# 9. Define Significance Thresholds
# ------------------------------------------------------------

p_value_threshold = 0.05
log2_fc_threshold = 1


# ------------------------------------------------------------
# 10. Identify Upregulated Genes
# ------------------------------------------------------------

upregulated = data[
    (data["Adjusted_P_Value"] < p_value_threshold) &
    (data["Log2_Fold_Change"] >= log2_fc_threshold)
]


# ------------------------------------------------------------
# 11. Identify Downregulated Genes
# ------------------------------------------------------------

downregulated = data[
    (data["Adjusted_P_Value"] < p_value_threshold) &
    (data["Log2_Fold_Change"] <= -log2_fc_threshold)
]

# %% Cell 4
# ------------------------------------------------------------
# 12. Display Differential Expression Results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("             DIFFERENTIAL EXPRESSION RESULTS")
print("=" * 65)

print("\nTotal genes analyzed:", len(data))

print(
    "Significantly Upregulated Genes:",
    len(upregulated)
)

print(
    "Significantly Downregulated Genes:",
    len(downregulated)
)


# ------------------------------------------------------------
# 13. Display Upregulated Genes
# ------------------------------------------------------------

print("\n--- UPREGULATED GENES ---")

if len(upregulated) > 0:

    result_columns = [
        "Gene",
        "Control_Mean",
        "Treatment_Mean",
        "Log2_Fold_Change",
        "P_Value",
        "Adjusted_P_Value"
    ]

    print(
        upregulated[result_columns]
        .sort_values("Adjusted_P_Value")
        .to_string(index=False)
    )

else:

    print("No significantly upregulated genes found.")


# ------------------------------------------------------------
# 14. Display Downregulated Genes
# ------------------------------------------------------------

print("\n--- DOWNREGULATED GENES ---")

if len(downregulated) > 0:

    result_columns = [
        "Gene",
        "Control_Mean",
        "Treatment_Mean",
        "Log2_Fold_Change",
        "P_Value",
        "Adjusted_P_Value"
    ]

    print(
        downregulated[result_columns]
        .sort_values("Adjusted_P_Value")
        .to_string(index=False)
    )

else:

    print("No significantly downregulated genes found.")

# %% Cell 5

# ------------------------------------------------------------
# 15. Display Complete Results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                  COMPLETE RESULTS")
print("=" * 65)

complete_columns = [
    "Gene",
    "Control_Mean",
    "Treatment_Mean",
    "Log2_Fold_Change",
    "P_Value",
    "Adjusted_P_Value"
]

print(
    data[complete_columns].to_string(index=False)
)


# ------------------------------------------------------------
# 16. Save Results
# ------------------------------------------------------------

data.to_csv(
    "differential_expression_results.csv",
    index=False
)

print("\nResults saved to:")
print("differential_expression_results.csv")

# %% Cell 6
# ------------------------------------------------------------
# 17. Volcano Plot
# ------------------------------------------------------------

data["Negative_Log10_P"] = -np.log10(
    data["Adjusted_P_Value"]
)

plt.figure(figsize=(10, 6))

plt.scatter(
    data["Log2_Fold_Change"],
    data["Negative_Log10_P"],
    alpha=0.7
)

# Fold-change thresholds

plt.axvline(
    x=1,
    linestyle="--"
)

plt.axvline(
    x=-1,
    linestyle="--"
)

# P-value threshold

plt.axhline(
    y=-np.log10(0.05),
    linestyle="--"
)

plt.xlabel("Log2 Fold Change")

plt.ylabel("-Log10 Adjusted P-Value")

plt.title("RNA-Seq Differential Gene Expression")

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 18. End of Program
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("              ANALYSIS COMPLETED")
print("=" * 65)
