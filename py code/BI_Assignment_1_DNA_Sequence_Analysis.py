"""Python code exported from BI_Assignment_1_DNA_Sequence_Analysis.ipynb."""

# %% Cell 1
# --------------------------------------------------
# 1. Function to validate DNA sequence
# --------------------------------------------------

def validate_dna(sequence):
    valid_bases = set("ATGC")

    for base in sequence:
        if base not in valid_bases:
            return False

    return True


# --------------------------------------------------
# 2. Function to calculate GC content
# --------------------------------------------------

def calculate_gc_content(sequence):
    gc_count = sequence.count("G") + sequence.count("C")
    gc_content = (gc_count / len(sequence)) * 100

    return gc_content


# --------------------------------------------------
# 3. Function to find motif positions
# --------------------------------------------------

def find_motif(sequence, motif):
    positions = []

    start = 0

    while True:
        position = sequence.find(motif, start)

        if position == -1:
            break

        # +1 because biological positions generally
        # start from 1 instead of 0
        positions.append(position + 1)

        start = position + 1

    return positions


# --------------------------------------------------
# 4. Function to identify coding regions / ORFs
# --------------------------------------------------

def find_orfs(sequence):
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]

    orfs = []

    # Check all three reading frames
    for frame in range(3):

        i = frame

        while i <= len(sequence) - 3:

            codon = sequence[i:i + 3]

            if codon == start_codon:

                start_position = i

                # Search for a stop codon
                j = i + 3

                while j <= len(sequence) - 3:

                    stop_codon = sequence[j:j + 3]

                    if stop_codon in stop_codons:

                        end_position = j + 3

                        orf_sequence = sequence[start_position:end_position]

                        orfs.append({
                            "start": start_position + 1,
                            "end": end_position,
                            "stop_codon": stop_codon,
                            "sequence": orf_sequence,
                            "length": len(orf_sequence)
                        })

                        # Move forward after finding ORF
                        i = j + 3
                        break

                    j += 3

                else:
                    # No stop codon found
                    i += 3

            else:
                i += 3

    return orfs


# --------------------------------------------------
# 5. Main Program
# --------------------------------------------------

print("=" * 50)
print("       DNA SEQUENCE ANALYSIS")
print("=" * 50)

# Input DNA sequence
sequence = input("Enter DNA sequence: ").strip().upper()

# Validate sequence
if not sequence:

    print("Error: DNA sequence cannot be empty.")

elif not validate_dna(sequence):

    print("Error: Invalid DNA sequence.")
    print("Only A, T, G and C are allowed.")

else:

    print("\nDNA sequence is valid.")
    print("Sequence length:", len(sequence))

    # --------------------------------------------------
    # GC Content
    # --------------------------------------------------

    gc_content = calculate_gc_content(sequence)

    print("\n--- GC CONTENT ---")
    print("GC Content: {:.2f}%".format(gc_content))

    # --------------------------------------------------
    # Motif Search
    # --------------------------------------------------

    motif = input("\nEnter motif to search: ").strip().upper()

    # Validate motif
    if not motif:
        print("Error: Motif cannot be empty.")

    elif not validate_dna(motif):
        print("Error: Invalid motif.")
        print("Motif should contain only A, T, G and C.")

    else:

        positions = find_motif(sequence, motif)

        print("\n--- MOTIF SEARCH ---")
        print("Motif:", motif)

        if positions:
            print("Motif found at positions:", positions)
            print("Number of occurrences:", len(positions))
        else:
            print("Motif not found.")

    # --------------------------------------------------
    # ORF Identification
    # --------------------------------------------------

    print("\n--- CODING REGION / ORF ANALYSIS ---")

    orfs = find_orfs(sequence)

    if orfs:

        print("Number of ORFs found:", len(orfs))

        for number, orf in enumerate(orfs, start=1):

            print("\nORF", number)
            print("Start position:", orf["start"])
            print("End position:", orf["end"])
            print("Stop codon:", orf["stop_codon"])
            print("Length:", orf["length"], "bp")
            print("Sequence:", orf["sequence"])

    else:

        print("No complete coding region (ORF) found.")

print("\n" + "=" * 50)
print("           ANALYSIS COMPLETED")
print("=" * 50)
#ATGCGTACGTTAGCTAGCTAGGCTAACGTTAGCGATCGATGCTAGCTAGGCTA
#GCTAGC
