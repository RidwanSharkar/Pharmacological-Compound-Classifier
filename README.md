**PHARMACOLOGICAL & CHEMICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Classification, Mechanisms of Action, and Therapeutic Potentials using Machine Learning to identify Structure-Activity-Relationships (SAR)
________________________________________________________________________________________________________________

OVERVIEW:
This project employs machine learning to identify the Structure-Activity Relationships (SAR) of chemical compounds based on their extensive molecular properties. Inspired by this dataset of 635 psychoactive compoounds and their molecular properties (117),
(https://www.kaggle.com/datasets/thedevastator/psychedelic-drug-database?resource=download)
a webscraper is set up to retrieve data of all compounds in PubChem.ncbi that possess a known 'MeSH Pharmacological Classification' (totaling 17,866 entries):
[Anti-Psychotic Agents, Serotonin Receptor Agonists, Dopamine Uptake Inhibitors, Protein Kinase Inhibitors, Vasoconstrictor Agents, etc]

RD-Kit (Cheminformatics) is then used to retrieve and compute all scraped compounds' molecular popularties (157), which is fed to a Random Forest/Gradiant Boosting/NLPClassifier machine learning model to determine which molecular properties correlate highest with the compounds' pharmacological action. These weighed attributes are then used by the model to predict the pharmacological classification of a novel compound whose 'MeSH Pharmacological Classification' is not listed on PubChem. 

________________________________________________________________________________________________________________
![test2](https://github.com/user-attachments/assets/c96fb577-adc7-4f75-87f3-fbc0345e2481)

________________________________________________________________________________________________________________

TECH STACK:

Frontend: React, TypeScript

Backend: Flask, Python

Cheminformatics Toolkit: RDKit

Data Storage: AWS S3, Parquet

Machine Learning: Scikit-learn

API: RESTful
