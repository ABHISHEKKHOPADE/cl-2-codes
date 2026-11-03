"""Python code exported from BI_Assignment_4_Molecular_Docking_Virtual_Screening.ipynb."""

# %% Cell 1
import os, shutil, numpy as np, pandas as pd, matplotlib.pyplot as plt
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE=True
    print("RDKit available")
except ImportError:
    RDKIT_AVAILABLE=False
    print("RDKit not installed. Install with: conda install -c conda-forge rdkit")

# %% Cell 2
ligands_df=pd.DataFrame({
    "Molecule_ID":["Ligand_01","Ligand_02","Ligand_03","Ligand_04","Ligand_05","Ligand_06"],
    "SMILES":["CCO","CC(=O)Oc1ccccc1C(=O)O","CCN(CC)CC","c1ccccc1",
              "CC(C)Cc1ccc(cc1)C(C)C(=O)O","CCOC(=O)c1ccccc1"]
})
display(ligands_df)

# %% Cell 3
if RDKIT_AVAILABLE:
    rows=[]
    for _,r in ligands_df.iterrows():
        m=Chem.MolFromSmiles(r.SMILES)
        if m:
            rows.append({
                "Molecule_ID":r.Molecule_ID,"SMILES":r.SMILES,
                "MolWt":Descriptors.MolWt(m),"LogP":Crippen.MolLogP(m),
                "HBD":Lipinski.NumHDonors(m),"HBA":Lipinski.NumHAcceptors(m),
                "TPSA":rdMolDescriptors.CalcTPSA(m),
                "RotatableBonds":Lipinski.NumRotatableBonds(m)})
    desc=pd.DataFrame(rows)
    display(desc)
    filtered=desc[(desc.MolWt<=500)&(desc.LogP<=5)&(desc.HBD<=5)&(desc.HBA<=10)]
    print("Candidates passing basic filter:",len(filtered),"/",len(desc))
    display(filtered)
else:
    print("Install RDKit to run descriptor calculations.")

# %% Cell 4
if RDKIT_AVAILABLE:
    ref=desc.iloc[0]
    cols=["MolWt","LogP","HBD","HBA","TPSA","RotatableBonds"]
    scale=(desc[cols].max()-desc[cols].min()).replace(0,1)
    d=[]
    for _,r in desc.iterrows():
        d.append(np.sqrt(np.sum(((r[cols].astype(float)-ref[cols].astype(float))/scale)**2)))
    desc["Descriptor_Distance"]=d
    desc["Screening_Score"]=1/(1+desc.Descriptor_Distance)
    screening=desc.sort_values("Screening_Score",ascending=False)
    display(screening[["Molecule_ID","Descriptor_Distance","Screening_Score"]])
    screening.plot(x="Molecule_ID",y="Screening_Score",kind="bar",legend=False)
    plt.title("Descriptor-Based Virtual Screening"); plt.ylabel("Score"); plt.tight_layout(); plt.show()

# %% Cell 5
vina=shutil.which("vina")
receptor="receptor.pdbqt"; ligand="ligand_01.pdbqt"
center_x,center_y,center_z=0.0,0.0,0.0
size_x,size_y,size_z=20.0,20.0,20.0
print("Vina:",vina or "NOT FOUND")
if vina and os.path.exists(receptor) and os.path.exists(ligand):
    cmd=[vina,"--receptor",receptor,"--ligand",ligand,
         "--center_x",str(center_x),"--center_y",str(center_y),"--center_z",str(center_z),
         "--size_x",str(size_x),"--size_y",str(size_y),"--size_z",str(size_z),
         "--out","docked_ligand.pdbqt","--log","docking.log"]
    print("Ready to execute:")
    print(" ".join(cmd))
else:
    print("Provide prepared PDBQT files, docking-box values, and AutoDock Vina to perform real docking.")
