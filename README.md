**PHARMACOLOGICAL & CHEMICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Classification, Mechanisms of Action, and Therapeutic Potentials using Machine Learning to identify Structure-Activity-Relationships (SAR) 

OVERVIEW:

Inspired by this dataset of 635 psychoactive compoounds and their molecular properties (117),

https://www.kaggle.com/datasets/thedevastator/psychedelic-drug-database?resource=download

a webscraper is set up to retrieve the data of all compounds in PubChem that possess a known 'MeSH Pharmacological Classification' (17,866), which each compound may contain multiple of: 

[Anti-Psychotic Agents, Serotonin Receptor Agonists, Dopamine Uptake Inhibitors, Protein Kinase Inhibitors, Vasoconstrictor Agents, etc]

RD-Kit (Cheminformatics) is then used to retrieve and compute all scraped compounds' molecular popularties (157), which is fed to a Random Forest/Gradiant Boosting/NLPClassifier machine learning model to determine which molecular properties correlate highest with the compounds' pharmacological action. 




**Tech Stack:**
Frontend: React, TypeScript
Backend: Flask, Python
Cheminformatics Toolkit: RDKit
Data Storage: AWS S3, Parquet
Machine Learning: Scikit-learn
RESTful API
