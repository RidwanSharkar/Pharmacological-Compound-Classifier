# backend/App.py

import pandas as pd
from flask import Flask, request, jsonify
import pyarrow.parquet as pq
import boto3
from io import BytesIO
from flask_cors import CORS
import ast
import joblib
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors, BondType
import requests

app = Flask(__name__)
CORS(app)

s3 = boto3.client('s3')
bucket_name = 'molecular-and-pharmacological-data'
file_key = 'parquet/compoundClassifications_scraped.parquet'

#reducedAccuracy MODEL FOR LOAD SIZE======================================================

model_key = 'Model2/RandomForestModel.pkl'
mlb_key = 'Model2/MultiLabelBinarizer.pkl'

def load_model_and_mlb():
    model_obj = s3.get_object(Bucket=bucket_name, Key=model_key)
    mlb_obj = s3.get_object(Bucket=bucket_name, Key=mlb_key)
    model = joblib.load(BytesIO(model_obj['Body'].read()))
    mlb = joblib.load(BytesIO(mlb_obj['Body'].read()))
    return model, mlb

model, mlb = load_model_and_mlb()


#==========================================================================================

def read_parquet_from_s3():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        if 'Body' in obj:
            parquet_file = BytesIO(obj['Body'].read())
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            print("Columns in DataFrame: ", df.columns)  
            print("DataFrame head: ", df.head()) 
            return df
        else:
            print("Error: 'Body' notfound in S3 object")
    except Exception as e:
        print(f"Failed to read from S3: {str(e)}")
    return pd.DataFrame() 

#==========================================================================================

@app.route('/activities', methods=['GET'])
def get_activities():
    try:
        df = read_parquet_from_s3()
        if df.empty:
            return jsonify({'error': 'No data found in Parquet'}), 500

        activity_counts = {}
        total_count = 0
        for activities_str in df['Activities'].dropna():
            try:
                activities = ast.literal_eval(activities_str)
                for activity in activities:
                    if activity in activity_counts:
                        activity_counts[activity] += 1
                    else:
                        activity_counts[activity] = 1
                    total_count += 1 
            except ValueError:
                continue

        activity_list = [{'activity': k, 'count': v} for k, v in sorted(activity_counts.items())]
        activity_list.append({'activity': 'Total', 'count': total_count})  # total
        return jsonify(activity_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#==========================================================================================    

@app.route('/search', methods=['POST'])
def search():
    content = request.json
    search_term = content.get('searchTerm')
    if not search_term:
        return jsonify({'error': 'No search term provided'}), 400
    try:
        df = read_parquet_from_s3()
        if df.empty:
            return jsonify({'error': 'No data found in Parquet'}), 500

        if search_term.isdigit():
            # CID SEARCH
            cid = int(search_term)
            result = df[df['CID'] == cid]

            if not result.empty:
                compound_name = result['Compound Name'].iloc[0] if 'Compound Name' in result.columns else 'N/A'
                activities = []
                if 'Activities' in result.columns:
                    activities_data = result['Activities'].iloc[0]
                    if isinstance(activities_data, str):
                        try:
                            activities = ast.literal_eval(activities_data)
                        except ValueError:
                            activities = []
                    elif pd.isna(activities_data):
                        activities = []
                response = {
                    'CID': cid,
                    'Compound Name': compound_name,
                    'Activities': activities
                }
                return jsonify(response)
            

            else:
                # IF *CID* NOT FOUND, ML PREDICT
                base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
                url = f"{base_url}/data/compound/{cid}/JSON"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                # Scraped Compound Name
                compound_name = data['Record']['RecordTitle']
                smiles = fetch_smiles(cid)
                if smiles:
                    descriptors = compute_descriptors(smiles)
                    if descriptors:
                        prediction = model.predict([list(descriptors.values())])
                        predicted_activities = mlb.inverse_transform(prediction)[0]
                        return jsonify({
                            'CID': cid,
                            'Compound Name': compound_name,
                            'Predicted Activities': list(predicted_activities)
                        })
                return jsonify({'error': 'CID not found or SMILES not retrievable'}), 404
            


        else:
            # COMPOUND NAME SEARCH
            compound_name_results = df[df['Compound Name'].str.contains(search_term, case=False, na=False)]
            
            if not compound_name_results.empty:
                response_data = compound_name_results[['CID', 'Compound Name']].to_dict(orient='records')
                return jsonify(response_data)
            else:
                # ACTIVITY SEARCH
                activity = search_term
                filtered_results = df[df['Activities'].apply(lambda x: activity in ast.literal_eval(x) if isinstance(x, str) else False)]
                if not filtered_results.empty:
                    response_data = filtered_results[['CID', 'Compound Name']].to_dict(orient='records')
                    return jsonify(response_data)
                else:
                    return jsonify({'error': 'No matching compounds or activities found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unexpected error'}), 500



#============================================================================================================================
def fetch_smiles(pubchem_cid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
    response = requests.get(url)
    data = response.json()
    if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(data['PropertyTable']['Properties']) > 0:
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    else:
        return None

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
        # EXPANDED SET FILTERED                                          
        'EState_VSA1': Descriptors.EState_VSA1(mol),                        
        'EState_VSA2': Descriptors.EState_VSA2(mol),
        'EState_VSA3': Descriptors.EState_VSA3(mol),
        'EState_VSA4': Descriptors.EState_VSA4(mol),
        'EState_VSA5': Descriptors.EState_VSA5(mol),
        'EState_VSA6': Descriptors.EState_VSA6(mol),
        'EState_VSA7': Descriptors.EState_VSA7(mol),
        'EState_VSA8': Descriptors.EState_VSA8(mol),
        'EState_VSA9': Descriptors.EState_VSA9(mol),
        'EState_VSA10': Descriptors.EState_VSA10(mol),
        #'EState_VSA11': Descriptors.EState_VSA11(mol),                      # electrotopological state (E-state)
        'VSA_EState1': Descriptors.VSA_EState1(mol),
        'VSA_EState2': Descriptors.VSA_EState2(mol),
        'VSA_EState3': Descriptors.VSA_EState3(mol),
        'VSA_EState4': Descriptors.VSA_EState4(mol),
        'VSA_EState5': Descriptors.VSA_EState5(mol),
        'VSA_EState6': Descriptors.VSA_EState6(mol),
        'VSA_EState7': Descriptors.VSA_EState7(mol),
        'VSA_EState8': Descriptors.VSA_EState8(mol),
        'VSA_EState9': Descriptors.VSA_EState9(mol),
        'VSA_EState10': Descriptors.VSA_EState10(mol),
        'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol),        # Maximum absolute EState index
        'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),              # Maximum EState index
        'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol),        # Minimum absolute EState index
        'MinEStateIndex': Descriptors.MinEStateIndex(mol),              # Minimum EState index
        'qed': Descriptors.qed(mol),                                    # Quantitative Estimate of Drug-likeness
        'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),              # Molecular weight of heavy atoms
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),    # Number of valence electrons
        #'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),    # Number of radical electrons
        'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),          # Maximum partial charge
        'MinPartialCharge': Descriptors.MinPartialCharge(mol),          # Minimum partial charge
        'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol),    # Maximum absolute partial charge
        'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(mol),    # Minimum absolute partial charge
        'FpDensityMorgan1': Descriptors.FpDensityMorgan1(mol),          # Fingerprint density (Morgan, radius 1)
        'FpDensityMorgan2': Descriptors.FpDensityMorgan2(mol),          # Fingerprint density (Morgan, radius 2)
        'FpDensityMorgan3': Descriptors.FpDensityMorgan3(mol),          # Fingerprint density (Morgan, radius 3)
        'BCUT2D_MWHI': Descriptors.BCUT2D_MWHI(mol),                    # BCUT2D descriptor using atomic mass (high)
        'BCUT2D_MWLOW': Descriptors.BCUT2D_MWLOW(mol),                  # BCUT2D descriptor using atomic mass (low)
        'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI(mol),                  # BCUT2D descriptor using atomic charge (high)
        'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO(mol),                  # BCUT2D descriptor using atomic charge (low)
        'BCUT2D_LOGPHI': Descriptors.BCUT2D_LOGPHI(mol),                # BCUT2D descriptor using atomic logP (high)
        'BCUT2D_LOGPLOW': Descriptors.BCUT2D_LOGPLOW(mol),              # BCUT2D descriptor using atomic logP (high)
        'BCUT2D_MRHI': Descriptors.BCUT2D_MRHI(mol),                    # BCUT2D descriptor using molar refractivity (high)
        'BCUT2D_MRLOW': Descriptors.BCUT2D_MRLOW(mol),                  # BCUT2D descriptor using molar refractivity (low)
        'AvgIpc': Descriptors.AvgIpc(mol),                              # Average information content (neighborhood symmetry of 0-5)***
        'BalabanJ': Descriptors.BalabanJ(mol),                          # Balaban's J index - topological descriptor of molecular shape
        'BertzCT': Descriptors.BertzCT(mol),                            # Bertz CT - a topological complexity index
        'SlogP': Crippen.MolLogP(mol),  # SlogP
        'SMR': Crippen.MolMR(mol),  # SMR
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),  # LabuteASA
        'TPSA': Descriptors.TPSA(mol),  # TPSA
        #'AMW': Descriptors.MolWt(mol),  # AMW
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

        'slogp_VSA1': rdMolDescriptors.SlogP_VSA_(mol)[0],
        'slogp_VSA2' : rdMolDescriptors.SlogP_VSA_(mol)[1],
        'slogp_VSA3' : rdMolDescriptors.SlogP_VSA_(mol)[2],
        'slogp_VSA4' : rdMolDescriptors.SlogP_VSA_(mol)[3],
        'slogp_VSA5' : rdMolDescriptors.SlogP_VSA_(mol)[4],
        'slogp_VSA6' : rdMolDescriptors.SlogP_VSA_(mol)[5],
        'slogp_VSA7' : rdMolDescriptors.SlogP_VSA_(mol)[6],
        'slogp_VSA8' : rdMolDescriptors.SlogP_VSA_(mol)[7],
        #'slogp_VSA9' : rdMolDescriptors.SlogP_VSA_(mol)[8],
        'slogp_VSA10' : rdMolDescriptors.SlogP_VSA_(mol)[9],
        'slogp_VSA11' : rdMolDescriptors.SlogP_VSA_(mol)[10],
        'slogp_VSA12' : rdMolDescriptors.SlogP_VSA_(mol)[11],

        'smr_VSA1' : rdMolDescriptors.SMR_VSA_(mol)[0],
        'smr_VSA2' : rdMolDescriptors.SMR_VSA_(mol)[1],
        'smr_VSA3' : rdMolDescriptors.SMR_VSA_(mol)[2],
        'smr_VSA4' : rdMolDescriptors.SMR_VSA_(mol)[3],
        'smr_VSA5' : rdMolDescriptors.SMR_VSA_(mol)[4],
        'smr_VSA6' : rdMolDescriptors.SMR_VSA_(mol)[5],
        'smr_VSA7' : rdMolDescriptors.SMR_VSA_(mol)[6],
        #'smr_VSA8' : rdMolDescriptors.SMR_VSA_(mol)[7],
        'smr_VSA9' : rdMolDescriptors.SMR_VSA_(mol)[8],
        'smr_VSA10' : rdMolDescriptors.SMR_VSA_(mol)[9],
        'peoe_VSA1' : rdMolDescriptors.PEOE_VSA_(mol)[0],
        'peoe_VSA2' : rdMolDescriptors.PEOE_VSA_(mol)[1],
        'peoe_VSA3' : rdMolDescriptors.PEOE_VSA_(mol)[2],
        'peoe_VSA4' : rdMolDescriptors.PEOE_VSA_(mol)[3],
        'peoe_VSA5' : rdMolDescriptors.PEOE_VSA_(mol)[4],
        'peoe_VSA6' : rdMolDescriptors.PEOE_VSA_(mol)[5],
        'peoe_VSA7' : rdMolDescriptors.PEOE_VSA_(mol)[6],
        'peoe_VSA8' : rdMolDescriptors.PEOE_VSA_(mol)[7],
        'peoe_VSA9' : rdMolDescriptors.PEOE_VSA_(mol)[8],
        'peoe_VSA10' : rdMolDescriptors.PEOE_VSA_(mol)[9],
        'peoe_VSA11' : rdMolDescriptors.PEOE_VSA_(mol)[10],
        'peoe_VSA12' : rdMolDescriptors.PEOE_VSA_(mol)[11],
        'peoe_VSA13' : rdMolDescriptors.PEOE_VSA_(mol)[12],
        'peoe_VSA14' : rdMolDescriptors.PEOE_VSA_(mol)[13],

        'MQN1' : rdMolDescriptors.MQNs_(mol)[0],
        'MQN2' : rdMolDescriptors.MQNs_(mol)[1],
        'MQN3' : rdMolDescriptors.MQNs_(mol)[2],
        #'MQN4' : rdMolDescriptors.MQNs_(mol)[3],
        #'MQN5' : rdMolDescriptors.MQNs_(mol)[4],
        'MQN6' : rdMolDescriptors.MQNs_(mol)[5],
        'MQN7' : rdMolDescriptors.MQNs_(mol)[6],
        'MQN8' : rdMolDescriptors.MQNs_(mol)[7],
        'MQN9' : rdMolDescriptors.MQNs_(mol)[8],
        'MQN10' : rdMolDescriptors.MQNs_(mol)[9],
        'MQN11' : rdMolDescriptors.MQNs_(mol)[10],
        'MQN12' : rdMolDescriptors.MQNs_(mol)[11],
        'MQN13' : rdMolDescriptors.MQNs_(mol)[12],
        'MQN14' : rdMolDescriptors.MQNs_(mol)[13],
        'MQN15' : rdMolDescriptors.MQNs_(mol)[14],
        'MQN16' : rdMolDescriptors.MQNs_(mol)[15],
        'MQN17' : rdMolDescriptors.MQNs_(mol)[16],
        #'MQN18' : rdMolDescriptors.MQNs_(mol)[17],
        'MQN19' : rdMolDescriptors.MQNs_(mol)[18],
        'MQN20' : rdMolDescriptors.MQNs_(mol)[19],
        'MQN21' : rdMolDescriptors.MQNs_(mol)[20],
        'MQN22' : rdMolDescriptors.MQNs_(mol)[21],
        'MQN23' : rdMolDescriptors.MQNs_(mol)[22],
        'MQN24' : rdMolDescriptors.MQNs_(mol)[23],
        'MQN25' : rdMolDescriptors.MQNs_(mol)[24],
        'MQN26' : rdMolDescriptors.MQNs_(mol)[25],
        'MQN27' : rdMolDescriptors.MQNs_(mol)[26],
        'MQN28' : rdMolDescriptors.MQNs_(mol)[27],
        'MQN29' : rdMolDescriptors.MQNs_(mol)[28],
        'MQN30' : rdMolDescriptors.MQNs_(mol)[29],
        'MQN31' : rdMolDescriptors.MQNs_(mol)[30],
        'MQN32' : rdMolDescriptors.MQNs_(mol)[31],
        'MQN33' : rdMolDescriptors.MQNs_(mol)[32],
        'MQN34' : rdMolDescriptors.MQNs_(mol)[33],
        'MQN35' : rdMolDescriptors.MQNs_(mol)[34],
        'MQN36' : rdMolDescriptors.MQNs_(mol)[35],
        'MQN37' : rdMolDescriptors.MQNs_(mol)[36],
        #'MQN38' : rdMolDescriptors.MQNs_(mol)[37],
        #'MQN39' : rdMolDescriptors.MQNs_(mol)[38],
        'MQN40' : rdMolDescriptors.MQNs_(mol)[39],
        'MQN41' : rdMolDescriptors.MQNs_(mol)[40],
        'MQN42' : rdMolDescriptors.MQNs_(mol)[41]
        }
    return base_descriptors
#============================================================================================================================




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)