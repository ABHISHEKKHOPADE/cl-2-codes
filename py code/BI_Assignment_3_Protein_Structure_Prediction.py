"""Python code exported from BI_Assignment_3_Protein_Structure_Prediction.ipynb."""

# %% Cell 1
import numpy as np

# ------------------------------------------------------------
# 1. Function to Validate Protein Sequence
# ------------------------------------------------------------

def validate_sequence(sequence):

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    for amino_acid in sequence:

        if amino_acid not in valid_amino_acids:
            return False

    return True


# ------------------------------------------------------------
# 2. Function to Calculate Sequence Identity
# ------------------------------------------------------------

def sequence_identity(target, template):

    length = min(len(target), len(template))

    matches = 0

    for i in range(length):

        if target[i] == template[i]:
            matches += 1

    identity = (matches / length) * 100

    return identity


# ------------------------------------------------------------
# 3. Function for Simple Sequence Alignment
# ------------------------------------------------------------

def simple_alignment(target, template):

    length = min(len(target), len(template))

    target_part = target[:length]
    template_part = template[:length]

    match_line = ""

    for i in range(length):

        if target_part[i] == template_part[i]:
            match_line += "|"

        else:
            match_line += "."

    return target_part, match_line, template_part

# %% Cell 2
# ------------------------------------------------------------
# 4. Function to Predict Secondary Structure
# ------------------------------------------------------------

def predict_secondary_structure(sequence):

    # Simplified classification for educational purposes

    helix_forming = set("ALMEKQRH")
    sheet_forming = set("VIFYWT")

    structure = ""

    for amino_acid in sequence:

        if amino_acid in helix_forming:

            structure += "H"

        elif amino_acid in sheet_forming:

            structure += "E"

        else:

            structure += "C"

    return structure


# ------------------------------------------------------------
# 5. Main Program
# ------------------------------------------------------------

print("=" * 65)
print("             PROTEIN STRUCTURE PREDICTION")
print("                 HOMOLOGY MODELING")
print("=" * 65)


# ------------------------------------------------------------
# Take Protein Sequence as Input
# ------------------------------------------------------------

sequence = input(
    "\nEnter the protein sequence: "
).strip().upper().replace(" ", "")


# ------------------------------------------------------------
# Validate Input
# ------------------------------------------------------------

if len(sequence) == 0:

    print("\nError: Protein sequence cannot be empty.")

    exit()


if not validate_sequence(sequence):

    print("\nError: Invalid protein sequence.")

    print(
        "Only the 20 standard amino acids "
        "are allowed."
    )

    exit()


print("\nProtein sequence is valid.")
print("Protein length:", len(sequence), "amino acids")

#MKTLLVAGAAAGTASAAPVLAADKPEMVRRLAEAGVEVVVEPGDNVVAMRDPQT

# %% Cell 3
# ------------------------------------------------------------
# Template Sequence
# ------------------------------------------------------------

# In real homology modeling, this template would be
# obtained from a structural database such as PDB.

template_sequence = (
    "MKTIIALSYIFCLVFADYK"
    "DDDDKGQRLIFAGKQLEDGRTLSDY"
    "VAVIDALNDSD"
)

template_name = "Template_Protein_1"


print("\n" + "=" * 65)
print("                 TEMPLATE INFORMATION")
print("=" * 65)

print("Template:", template_name)

print(
    "Template length:",
    len(template_sequence),
    "amino acids"
)


# ------------------------------------------------------------
# Calculate Sequence Identity
# ------------------------------------------------------------

identity = sequence_identity(
    sequence,
    template_sequence
)

print(
    "\nSequence Identity: {:.2f}%".format(identity)
)


# ------------------------------------------------------------
# Perform Simple Alignment
# ------------------------------------------------------------

target_part, match_line, template_part = simple_alignment(
    sequence,
    template_sequence
)


print("\n" + "=" * 65)
print("                    SEQUENCE ALIGNMENT")
print("=" * 65)

print("\nTarget   :", target_part)

print("           ", match_line)

print("Template :", template_part)

# %% Cell 4
# ------------------------------------------------------------
# Predict Secondary Structure
# ------------------------------------------------------------

secondary_structure = predict_secondary_structure(
    sequence
)


print("\n" + "=" * 65)
print("              PREDICTED SECONDARY STRUCTURE")
print("=" * 65)

print("\nH = Alpha Helix")
print("E = Beta Sheet")
print("C = Coil")

print("\nProtein Sequence:")
print(sequence)

print("\nPredicted Structure:")
print(secondary_structure)


# ------------------------------------------------------------
# Estimate Modeling Confidence
# ------------------------------------------------------------

if identity >= 50:

    confidence = "High"

elif identity >= 30:

    confidence = "Moderate"

else:

    confidence = "Low"


print("\nEstimated Modeling Confidence:", confidence)

# %% Cell 5

# ------------------------------------------------------------
# Structural Representation
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("             STRUCTURAL REPRESENTATION")
print("=" * 65)

print("\nPosition  Amino Acid  Structure")

print("-" * 35)


for i, amino_acid in enumerate(sequence):

    structure = secondary_structure[i]

    if structure == "H":

        symbol = "Alpha Helix"

    elif structure == "E":

        symbol = "Beta Sheet"

    else:

        symbol = "Coil"

    print(
        "{:8d}  {:11s}  {}".format(
            i + 1,
            amino_acid,
            symbol
        )
    )

# %% Cell 6
# ------------------------------------------------------------
# Final Result
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                    FINAL RESULT")
print("=" * 65)

print(
    "\nTarget Protein Length:",
    len(sequence),
    "amino acids"
)

print("Template:", template_name)

print(
    "Sequence Identity: {:.2f}%".format(identity)
)

print("Modeling Confidence:", confidence)

print("\nProtein structure prediction completed.")

print("=" * 65)
