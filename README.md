**PHARMACOLOGICAL & CHEMICAL COMPOUND CLASSIFIER**
- Predictive Modeling of Chemical Compounds' Classification, Mechanisms of Action, and Therapeutic Potentials using Machine Learning on Structure-Activity-Relationship (SAR)

OVERVIEW:


FEATURES:

Chemical Compound Search: Allows users to enter a PubChem Compound ID (CID) and retrieve chemical properties and predicted pharmacological classifications.
Machine Learning Integration: Utilizes a RandomForest model to predict potential therapeutic categories based on molecular descriptors extracted from compound structures.
Interactive Frontend: A React-based user interface that communicates with a Flask backend to process queries and display results.
Chemical Descriptor Calculation: Uses RDKit to compute molecular descriptors which are crucial for the machine learning model to make predictions.
Technology Stack
Frontend: React, TypeScript
Backend: Flask, Python
Chemistry Toolkit: RDKit
Data Storage: AWS S3, Parquet
Machine Learning: Scikit-learn, Joblib for model serialization
API: Flask provides a RESTful API that interacts with the React frontend.
