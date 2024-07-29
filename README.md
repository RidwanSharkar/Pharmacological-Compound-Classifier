________________________________________________________________________________________________________________
**PHARMACOLOGICAL & CHEMICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Classification, Mechanisms of Action, and Therapeutic Potentials using Machine Learning to identify Structure-Activity-Relationships (SAR)
________________________________________________________________________________________________________________

**OVERVIEW:**

This project employs Machine Learning to identify the Quantitative Structure-Activity Relationships (QSAR) of chemical compounds based on their extensive molecular properties. Inspired by this dataset of 635 psychoactive compoounds and their molecular properties (117)
[https://www.kaggle.com/datasets/thedevastator/psychedelic-drug-database?resource=download], 
a webscraper is set up to retrieve data from all compounds listed under PubChem.ncbi that possess a known 'MeSH Pharmacological Classification' (totaling 17,866 entries) to create and store a custom expanded dataset.


[Anti-Psychotic Agents, Serotonin Receptor Agonists, Dopamine Uptake Inhibitors, Protein Kinase Inhibitors, Vasoconstrictor Agents, etc]

RD-Kit (Cheminformatics) is then used to retrieve and compute all scraped compounds' molecular popularties (157), which is fed to a Random Forest/Gradiant Boosting/NLP-Classifier machine learning model to determine which molecular properties correlate highest with the compounds' pharmacological action. These weighed attributes are then used by the model to predict the pharmacological classification of a novel compound whose official 'MeSH Pharmacological Classification' is not listed on PubChem. 

________________________________________________________________________________________________________________
Hamming Loss - Multi-Label Classification - is utilized to allow independent prediction of each activity within the set of pharmacological activities:

![test2](https://github.com/user-attachments/assets/c96fb577-adc7-4f75-87f3-fbc0345e2481)
________________________________________________________________________________________________________________

**MOLECULAR PROPERTIES:**


• 'SlogP' - represents the compound's Lipophilicity, predicting how well it can cross cell membranes.

• 'SMR' - Molar Refractivity, representing molecule's volume and polarizability (measure of electron cloud's ability to distort)

• 'LabuteASA' - approx Surface Area

• 'TPSA' - Topological Polar Surface Area  (important for drug bioavailability)

• 'AMW' -  Average Molecular Weight of the compound.

• 'FractionCSP3' - fraction of carbons that are sp3 hybridized (indicating more Hydrogen saturation and flexibility of the molecule)

• 'EState_VSA1', ... , 'EState_VSA10' - for Electrotopological State of the molecule

• 'fr_alkyl_halide', 'fr_amide', 'fr_benzene', etc. - Functional Group count, which correlate chemical behavior and biological receptor activity.

• 'NumLipinskiHBA' - Number of Hydrogen Bond Acceptors as defined by Lipinski's rule of five (NO Count)

• 'NumLipinskiHBD' - Number of Hydrogen Bond Donors as defined by Lipinski's rule of five (NHOH Count)

• 'NumRotatableBonds' - Number of bonds in a molecule that can freely rotate.

• 'NumHeteroAtoms' - Number of atoms other than carbon and hydrogen (e.g., oxygen, nitrogen)

• 'NumAromaticRings', Number of aromatic ring structures (rings with alternating double bonds).

• 'NumSaturatedRings', Number of saturated rings (rings that contain single bonds only).

• 'NumAliphaticRings' Number of non-aromatic rings.

• 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v' - Chi connectivity index in valence state (measure of molecular branching).

• 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n' - Different orders of the Chi connectivity index, non-valence state.

• 'HallKierAlpha' - measure of molecular shape and electron distribution.

• 'kappa1', 'kappa2', 'kappa3' -  Kappa shape indices reflect molecular shape, symmetry, and branching.

• 'peoe_VSA1', 'peoe_VSA2', ... , 'peoe_VSA14' - relate to partial equalization of orbital electronegativity (PEOE).

• 'MQN1', 'MQN2', ... , 'MQN42' - Molecular Quantum Numbers, series of 42 ints capturing molecular structure information.

... + 100


________________________________________________________________________________________________________________
**TECH STACK:**

Frontend: React, TypeScript

Backend: Flask, Python

Cheminformatics Toolkit: RDKit

Data Storage: AWS S3, Parquet

Machine Learning: Scikit-learn

________________________________________________________________________________________________________________
