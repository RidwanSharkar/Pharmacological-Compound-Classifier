# backend\App.py:

import pandas as pd
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import joblib
import requests
from s3_services import s3_client, parquet_handler
from rdkit.Chem.rdchem import BondType
from flask import Flask, request, jsonify
from flask_cors import CORS
#============================================================================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

@app.route('/upload', methods=['POST'])
def upload_to_s3():
    # Data is sent as a CSV file and converted to Parquet
    df = pd.read_csv(request.files.get('file'))
    parquet_file = 'temp.parquet'
    parquet_handler.write_df_to_parquet(df, parquet_file)
    s3_client.upload_file_to_s3(parquet_file, 'molecular-and-pharmacological-data')
    return jsonify({'message': 'File uploaded successfully'})

# Load the RFmodel and the MultiLabelBinarizer
model = joblib.load('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\RandomForestModel.pkl')
mlb = joblib.load('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\MultiLabelBinarizer.pkl')
app = Flask(__name__)

#============================================================================================================================
def fetch_smiles(pubchem_cid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
    response = requests.get(url)
    data = response.json()
    if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(data['PropertyTable']['Properties']) > 0:
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    else:
        return None
#============================================================================================================================
def compute_vsa_descriptors(mol):
    calculated_descriptors = {}
    smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
    for i, value in enumerate(smr_vsa):
        calculated_descriptors[f'smr_VSA{i + 1}'] = value
    slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
    for i, value in enumerate(slogp_vsa):
        calculated_descriptors[f'slogp_VSA{i + 1}'] = value
    peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
    for i, value in enumerate(peoe_vsa):
        calculated_descriptors[f'peoe_VSA{i + 1}'] = value
    return calculated_descriptors

def compute_mqn_descriptors(mol):
    mqn_descriptors = rdMolDescriptors.MQNs_(mol)
    return {f'MQN{i + 1}': val for i, val in enumerate(mqn_descriptors)}

def count_amide_bonds(mol):
    amide_bond_count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == BondType.SINGLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetAtomicNum() == 6 and end_atom.GetAtomicNum() == 7) or \
                    (begin_atom.GetAtomicNum() == 7 and end_atom.GetAtomicNum() == 6):
                if any(n.GetAtomicNum() == 8 and n.GetTotalValence() == 2 for n in begin_atom.GetNeighbors()) or \
                        any(n.GetAtomicNum() == 8 and n.GetTotalValence() == 2 for n in end_atom.GetNeighbors()):
                    amide_bond_count += 1
    return amide_bond_count
#============================================================================================================================
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    base_descriptors = {
        'SlogP': Crippen.MolLogP(mol),  # SlogP
        'SMR': Crippen.MolMR(mol),  # SMR
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),  # LabuteASA
        'TPSA': Descriptors.TPSA(mol),  # TPSA
        'AMW': Descriptors.MolWt(mol),  # AMW
        'ExactMW': Descriptors.ExactMolWt(mol),  # ExactMW
        'NumLipinskiHBA': Lipinski.NOCount(mol),  # Updated to use Lipinski NOCount
        'NumLipinskiHBD': Lipinski.NHOHCount(mol),  # Updated to use Lipinski NHOHCount
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # NumRotatableBonds
        'NumHBD': Descriptors.NumHDonors(mol),  # NumHBD
        'NumHBA': Descriptors.NumHAcceptors(mol),  # NumHBA
        'NumAmideBonds': count_amide_bonds(mol),
        'NumHeteroAtoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),  # NumHeavyAtoms
        'NumAtoms': mol.GetNumAtoms(),  # ***NumAtoms EXCLUDES HYDROGENS***
        'NumRings': rdMolDescriptors.CalcNumRings(mol),  # NumRings
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),  # NumAromaticRings
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),  # NumSaturatedRings
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),  # NumAliphaticRings
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(mol),  # NumAromaticHeterocycles
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),  # NumSaturatedHeterocycles
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),  # NumAliphaticHeterocycles
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),  # NumAromaticCarbocycles
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),  # NumSaturatedCarbocycles
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),  # NumAliphaticCarbocycles
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),  # FractionCSP3
        'Chi0v': rdMolDescriptors.CalcChi0v(mol),
        'Chi1v': rdMolDescriptors.CalcChi1v(mol),
        'Chi2v': rdMolDescriptors.CalcChi2v(mol),
        'Chi3v': rdMolDescriptors.CalcChi3v(mol),
        'Chi4v': rdMolDescriptors.CalcChi4v(mol),       # Chi x9
        'Chi1n': rdMolDescriptors.CalcChi1n(mol),
        'Chi2n': rdMolDescriptors.CalcChi2n(mol),
        'Chi3n': rdMolDescriptors.CalcChi3n(mol),
        'Chi4n': rdMolDescriptors.CalcChi4n(mol),
        'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(mol),  # HallKierAlpha
        'kappa1': rdMolDescriptors.CalcKappa1(mol),     # kappa1
        'kappa2': rdMolDescriptors.CalcKappa2(mol),     # kappa2
        'kappa3': rdMolDescriptors.CalcKappa3(mol),     # kappa3 = 39
    }
    base_descriptors.update(compute_vsa_descriptors(mol))
    base_descriptors.update(compute_mqn_descriptors(mol))
    return base_descriptors
#============================================================================================================================

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    cid = content.get('cid')
    smiles = fetch_smiles(cid)
    if smiles:
        descriptors = compute_descriptors(smiles)
        if descriptors:
            df = pd.DataFrame([descriptors])
            prediction = model.predict(df)
            predicted_labels = mlb.inverse_transform(prediction)
            return jsonify({'CID': cid, 'SMILES': smiles, 'Predicted Activities': predicted_labels})
        else:
            return jsonify({'error': 'Could not compute descriptors from SMILES.'}), 400
    else:
        return jsonify({'error': 'SMILES could not be fetched with given CID.'}), 404
    

    


if __name__ == '__main__':
    app.run(debug=True)
