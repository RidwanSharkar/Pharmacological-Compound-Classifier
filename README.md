________________________________________________________________________________________________________________
**PHARMACOLOGICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Drug Classifications
- Using Machine Learning to identify Structure-Activity-Relationships (QSAR)
________________________________________________________________________________________________________________

**OVERVIEW:**

• This project employs various machine learning models to identify the Quantitative Structure-Activity Relationships (QSAR) of chemical compounds based on their extensive molecular properties. 

• Users can search and sort the database for pharmacological classifications & compounds - Compounds that DO NOT have any classification data available on PubChem will predicted by the model, if possible. 

________________________________________________________________________________________________________________

![image](https://github.com/user-attachments/assets/3792cedb-e24c-43db-8b43-d42e24049459)

![image](https://github.com/user-attachments/assets/4658cd73-f9e4-43e6-a2d0-feae73471db9)

________________________________________________________________________________________________________________

![image](https://github.com/user-attachments/assets/33332207-f3d2-4048-9620-31d982d3de61)

![image](https://github.com/user-attachments/assets/ecb1c8cc-8aac-44f6-9948-90d5415854d6)


________________________________________________________________________________________________________________

**METHODS 1:**

• Inspired by this dataset of 635 psychoactive compounds and their 117 molecular properties. 
[https://www.kaggle.com/datasets/thedevastator/psychedelic-drug-database?resource=download], 
a webscraper is set up to retrieve data from all compounds listed under PubChem.ncbi that possess a known 'MeSH Pharmacological Classification' (totaling 17,866 entries) to create and store a more expansive dataset.

• RD-Kit (Cheminformatics) is then used to retrieve and compute all scraped compounds' molecular popularties, which have been expanded to include 244 molecular porperties - now including counts of all functional groups (amides, halogens, ketones, etc.) as well as electrotopological states (E-state) information that directly correlate molecular behavior with biological receptor activity.

• All scraped and computed data is then fed to a Random Forest machine learning model to determine which molecular properties correlate highest with the compounds' pharmacological action. These weighed attributes are then used by the model to predict the pharmacological classification of a novel compound whose official 'MeSH Pharmacological Classification' is NOT listed on PubChem. 

________________________________________________________________________________________________________________

**Example 'MeSH Pharmacological Classification':**

• [Anti-Psychotic Agents, Serotonin Receptor Agonists, Dopamine Uptake Inhibitors, Protein Kinase Inhibitors, Vasoconstrictor Agents, etc.]

• Hamming Loss - Multi-Label Classification - is utilized to allow independent prediction of each activity within the set of pharmacological activities:

![image](https://github.com/user-attachments/assets/5289915e-480f-4cd7-8765-6997cde6edeb)


________________________________________________________________________________________________________________

**METHODS 2:** DEEP LEARNING MODEL

([https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422)) <br>

Feature Selection from Random Forest model is used to guide setup of a Deep Neural Network (DNN) with the following parameters to achieve highest validation accuracy and AUC:

"A higher AUC value corresponds to better prediction, and the AUC value is always more than zero and less than 100 %" <br>
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5297997/)

• Normalization of Molecular Properties using StandardScalar() (scaling to unit variance) 

• 1st Layer: 1760 Neuron Count

• 2nd Layer: 1240 Neuron Count

• 3rd Layer: 512 Neuron Count

• Output layer: Sigmoid Activation (for independent label predictions)  |  Dropout Layers: 0.25-0.4 

• Batch Size: 64  |  Epochs: 120

• Loss Function: Binary Crossentropy (to be compatible with Multi-Label-Classification setup)

• Learning Rate: 0.00035 with Adam Optimizer 

• Gradient Clipping & Early Stopping 

• BatchNormalization() & L2 Regularizers*

________________________________________________________________________________________________________________

![image](https://github.com/user-attachments/assets/38c7c6ba-6d76-4152-a24e-d227fc59c10b)


________________________________________________________________________________________________________________

**METHODS 3:**

To enhance model prediction accuracy for Psychoactive Compounds in particular, the original dataset of 17,866 compounds was pruned down to 5,107 compounds that ONLY include information relevant to Neurotransmitter Receptors within their Classifications: 

• Adenosine A1/2 Receptor Agonists<br>
• Adenosine A1/2 Receptor Antagonists<br>
• Cholinergic Agonists<br>
• Cholinergic Antagonists<br>
• Nicotinic Agonists<br>
• Nicotinic Antagonists<br>
• Muscarinic Agonists<br>
• Muscarinic Antagonists<br>
• GABA-A Receptor Agonists<br>
• GABA-A Receptor Antagonists<br>
• GABA-B Receptor Agonists<br>
• GABA-B Receptor Antagonists<br>
• Serotonin 5-HT1 Receptor Agonists<br>
• Serotonin 5-HT1 Receptor Antagonists<br>
• Serotonin 5-HT2 Receptor Agonists<br>
• Serotonin 5-HT2 Receptor Antagonists<br>
• Selective Serotonin Reuptake Inhibitors<br>
• Dopamine Agonists<br>
• Dopamine Antagonists<br>
• Dopamine Uptake Inhibitors<br>
• Cannabinoid Receptor Agonists<br>
• Cannabinoid Receptor Antagonists<br>
• Nootropic Agents<br>
• Neuroprotective Agents<br>
• Monoamine Oxidase Inhibitors<br>
• Psychotropic Drugs<br>
• Antipsychotic Agents<br>
• Adrenergic beta-1 Receptor Agonists<br>
• Adrenergic beta-1 Receptor Antagonists<br>
• Adrenergic alpha-2 Receptor Agonists<br>
• Adrenergic alpha-2 Receptor Antagonists<br>
• Histamine H1 Antagonists<br>
• Histamine H2 Antagonists<br>
• ... (119 Categories Total)
________________________________________________________________________________________________________________

**Example Molecular Properties:**

• 'SlogP' - represents the compound's Lipophilicity, predicting how well it can cross cell membranes<br>
• 'SMR' - Molar Refractivity, representing molecule's volume and polarizability (measure of electron cloud's ability to distort)<br>
• 'TPSA' - Topological Polar Surface Area  (important for drug bioavailability)<br>
• 'FractionCSP3' - fraction of carbons that are sp3 hybridized (indicating more Hydrogen saturation and flexibility of the molecule)<br>
• 'EState_VSA1', ... , 'EState_VSA10' - for Electrotopological State of the molecule<br>
• 'fr_alkyl_halide', 'fr_amide', 'fr_halogen', 'fr_Imine', etc. - Functional Group counts<br>
• 'NumLipinskiHBA' - Number of Hydrogen Bond Acceptors as defined by Lipinski's rule of five (NO Count)<br>
• 'NumLipinskiHBD' - Number of Hydrogen Bond Donors as defined by Lipinski's rule of five (NHOH Count)<br>
• 'NumRotatableBonds' - Number of bonds in a molecule that can freely rotate<br>
• 'NumHeteroAtoms' - Number of atoms other than carbon and hydrogen (e.g., oxygen, nitrogen)<br>
• 'NumAromaticRings', Number of aromatic ring structures (rings with alternating double bonds)<br>
• 'NumSaturatedRings', Number of saturated rings (rings that contain single bonds only - H saturation)<br>
• 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v' - Chi connectivity index in valence state (measure of molecular branching)<br>
• 'HallKierAlpha' - measure of molecular shape and electron distribution<br>
• 'kappa1', 'kappa2', 'kappa3' -  Kappa shape indices reflect molecular shape, symmetry, and branching<br>
• 'peoe_VSA1', 'peoe_VSA2', ... , 'peoe_VSA14' - relate to partial equalization of orbital electronegativity (PEOE)<br>
• 'MQN1', 'MQN2', ... , 'MQN42' - Molecular Quantum Numbers, series of 42 ints capturing molecular structure information<br>
• ... (244 Propeties Total)

[https://cadaster.eu/sites/cadaster.eu/files/challenge/descr.htm]

________________________________________________________________________________________________________________
**TECH STACK:**

**Frontend:** React, TypeScript

**Backend:** Flask, Python, RESTful API

**Data Storage:** AWS S3, Parquet

**Machine Learning:** TensorFlow, Keras, Optuna, Scikit-learn

