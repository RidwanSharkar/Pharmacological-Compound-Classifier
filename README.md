________________________________________________________________________________________________________________
**PHARMACOLOGICAL & CHEMICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Classifications and Therapeutic Potentials
- Using Machine Learning to identify Structure-Activity-Relationships (SAR)
________________________________________________________________________________________________________________

**OVERVIEW:**
• This project employs machine learning to identify the Quantitative Structure-Activity Relationships (QSAR) of chemical compounds based on their extensive molecular properties. 
![image](https://github.com/user-attachments/assets/a6ec329a-c907-4811-92f9-bd8823e5d0b4)
![Screenshot 2024-08-09 132916](https://github.com/user-attachments/assets/d33657c3-0700-42d5-bd8b-11eb557b3eac)

• Users can search and sort the database for classifications and compounds - Classifications that are NOT listed on PubChem will predicted by the model, if possible. 
![Screenshot 2024-08-09 132153](https://github.com/user-attachments/assets/f93ea499-dae5-4680-945b-c931e4d41072)
![Screenshot 2024-08-09 132745](https://github.com/user-attachments/assets/46e86eea-333d-4350-87eb-781ea396024e)



________________________________________________________________________________________________________________

**METHODS 1:**

• Inspired by this dataset of 635 psychoactive compoounds and their 117 molecular properties. 
[https://www.kaggle.com/datasets/thedevastator/psychedelic-drug-database?resource=download], 
a webscraper is set up to retrieve data from all compounds listed under PubChem.ncbi that possess a known 'MeSH Pharmacological Classification' (totaling 17,866 entries) to create and store a custom expanded dataset.

• RD-Kit is then used to retrieve and compute all scraped compounds' molecular popularties, which have been expanded to include 244 molecular porperties - now including counts of all functional groups (amides, halogens, ketones, etc.) as well as electrotopological states (E-state) information that directly correlate molecular behavior with biological receptor activity.

• All scraped and computed data is then fed to a Random Forest machine learning model to determine which molecular properties correlate highest with the compounds' pharmacological action. These weighed attributes are then used by the model to predict the pharmacological classification of a novel compound whose official 'MeSH Pharmacological Classification' is NOT listed on PubChem. 

________________________________________________________________________________________________________________

**Example 'MeSH Pharmacological Classification':**

[Anti-Psychotic Agents, Serotonin Receptor Agonists, Dopamine Uptake Inhibitors, Protein Kinase Inhibitors, Vasoconstrictor Agents, etc.]

Hamming Loss - Multi-Label Classification - is utilized to allow independent prediction of each activity within the set of pharmacological activities:

![1](https://github.com/user-attachments/assets/5f151664-3c4a-40bf-be7c-adf9fece44b5)
![2](https://github.com/user-attachments/assets/a55920cb-d1cf-4c30-a347-6b709b1cd671)



________________________________________________________________________________________________________________

**METHODS 2:**

([https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422)) Deep Neural Network (DNN) is set up with the following parameters:

• Normalization of Molecular Properties using StandardScalar() (scaling to unit variance) 

• 1st Layer: 1760 Neuron Count

• 2nd Layer: 1024 Neuron Count

• 3rd Layer: 512 Neuron Count

• Output layer: Sigmoid Activation (for independent label predictions)  |  Dropout Layers: 0.3-0.4 

• Batch Size: 64  |  Epochs: 120

• Loss Function: Binary Crossentropy (to be compatible with Multi-Label-Classification setup)

• Learning Rate: 0.00035 with Adam Optimizer 

• Gradient Clipping & Early Stopping 

• BatchNormalization() and L2 Regularizers were noted to reduce accuracy for this dataset:

DNN currently undergoing Optuna Trials for hyperparameter optimization (Testing combinations of the above parameters to output the the highest model accuracy) 

________________________________________________________________________________________________________________

**MOLECULAR PROPERTIES:**


• 'SlogP' - represents the compound's Lipophilicity, predicting how well it can cross cell membranes.

• 'SMR' - Molar Refractivity, representing molecule's volume and polarizability (measure of electron cloud's ability to distort)

• 'LabuteASA' - approx Surface Area

• 'TPSA' - Topological Polar Surface Area  (important for drug bioavailability)

• 'AMW' -  Average Molecular Weight of the compound.

• 'FractionCSP3' - fraction of carbons that are sp3 hybridized (indicating more Hydrogen saturation and flexibility of the molecule)

• 'EState_VSA1', ... , 'EState_VSA10' - for Electrotopological State of the molecule

• 'fr_alkyl_halide', 'fr_amide', 'fr_halogen', 'fr_Imine', etc. - Functional Group counts, which correlate chemical behavior to potential receptor activition.

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

• ... (244 Total)

[https://cadaster.eu/sites/cadaster.eu/files/challenge/descr.htm]

________________________________________________________________________________________________________________
**TECH STACK:**

Frontend: React, TypeScript

Backend: Flask, Python

Data Storage: AWS S3, Parquet

Machine Learning: TensorFlow, Keras, Optuna, Scikit-learn

RESTful API
