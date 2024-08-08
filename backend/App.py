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

#==========================================================================================

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
                compound_name = 'N/A'
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
        # EXPANDED SET                                          COUNTS OF MOLECULES' FUNCTIONAL GROUPS:
        'fr_Al_COO': Descriptors.fr_Al_COO(mol),                    # aliphatic carboxylic acids
        'fr_Al_OH': Descriptors.fr_Al_OH(mol),                      # aliphatic hydroxyl groups
        'fr_Al_OH_noTert': Descriptors.fr_Al_OH_noTert(mol),        # aliphatic hydroxyl groups excluding tertiary-OH
        'fr_ArN': Descriptors.fr_ArN(mol),                          # N functional groups on aromatic rings
        'fr_Ar_N': Descriptors.fr_Ar_N(mol),                        # aromatic N atoms
        'fr_Ar_NH': Descriptors.fr_Ar_NH(mol),                      # aromatic amines
        'fr_Ar_OH': Descriptors.fr_Ar_OH(mol),                      # aromatic hydroxyl groups
        'fr_COO': Descriptors.fr_COO(mol),                          # carboxylic acids
        'fr_COO2': Descriptors.fr_COO2(mol),                        # carboxylic acid derivatives
        'fr_C_O': Descriptors.fr_C_O(mol),                          # carbonyl O
        'fr_C_O_noCOO': Descriptors.fr_C_O_noCOO(mol),              # carbonyl O, excluding COOH
        #'fr_C_S': Descriptors.fr_C_S(mol),                          # thiocarbonyl
        'fr_HOCCN': Descriptors.fr_HOCCN(mol),                      # C(OH)CCN substructures
        'fr_Imine': Descriptors.fr_Imine(mol),                      # imines
        'fr_NH0': Descriptors.fr_NH0(mol),                          # tertiary amines
        'fr_NH1': Descriptors.fr_NH1(mol),                          # secondary amines
        'fr_NH2': Descriptors.fr_NH2(mol),                          # primary amines
        #'fr_N_O': Descriptors.fr_N_O(mol),                          # N-O bonds
        'fr_Ndealkylation1': Descriptors.fr_Ndealkylation1(mol),    # Count of XCCNR groups
        'fr_Ndealkylation2': Descriptors.fr_Ndealkylation2(mol),    # Count of tert-alicyclic amines
        'fr_Nhpyrrole': Descriptors.fr_Nhpyrrole(mol),              # Count of H-pyrrole nitrogens
        #'fr_SH': Descriptors.fr_SH(mol),                            # Count of thiol groups
        #'fr_aldehyde': Descriptors.fr_aldehyde(mol),                # Count of aldehydes
        #'fr_alkyl_carbamate': Descriptors.fr_alkyl_carbamate(mol),  # Count of alkyl carbamates
        #'fr_alkyl_halide': Descriptors.fr_alkyl_halide(mol),        # Count of alkyl halides
        'fr_allylic_oxid': Descriptors.fr_allylic_oxid(mol),        # Count of allylic oxidation sites
        'fr_amide': Descriptors.fr_amide(mol),                      # Count of amides
        #'fr_amidine': Descriptors.fr_amidine(mol),                  # Count of amidines
        'fr_aniline': Descriptors.fr_aniline(mol),                  # Count of anilines
        'fr_aryl_methyl': Descriptors.fr_aryl_methyl(mol),          # Count of aryl methyl sites
        #'fr_azide': Descriptors.fr_azide(mol),                      # Count of azides
        #'fr_azo': Descriptors.fr_azo(mol),                          # Count of azo groups
        'fr_benzene': Descriptors.fr_benzene(mol),                  # Count of benzene rings
        'fr_bicyclic': Descriptors.fr_bicyclic(mol),                # Count of bicyclic structures
        #'fr_diazo': Descriptors.fr_diazo(mol),                      # Count of diazo groups
        #'fr_dihydropyridine': Descriptors.fr_dihydropyridine(mol),  # Count of dihydropyridines
        'fr_epoxide': Descriptors.fr_epoxide(mol),                  # Count of epoxide rings
        'fr_ester': Descriptors.fr_ester(mol),                      # Count of esters
        'fr_ether': Descriptors.fr_ether(mol),                      # Count of ether groups
        #'fr_furan': Descriptors.fr_furan(mol),                      # Count of furan rings
        'fr_guanido': Descriptors.fr_guanido(mol),                  # Count of guanidine groups
        'fr_halogen': Descriptors.fr_halogen(mol),                  # Count of halogens
        #'fr_hdrzine': Descriptors.fr_hdrzine(mol),                  # Count of hydrazine groups
        #'fr_hdrzone': Descriptors.fr_hdrzone(mol),                  # Count of hydrazone groups
        #'fr_imidazole': Descriptors.fr_imidazole(mol),              # Count of imidazole rings
        #'fr_imide': Descriptors.fr_imide(mol),                      # Count of imide groups
        #'fr_isocyan': Descriptors.fr_isocyan(mol),                  # Count of isocyanates
        #'fr_isothiocyan': Descriptors.fr_isothiocyan(mol),          # Count of isothiocyanates
        'fr_ketone': Descriptors.fr_ketone(mol),                    # Count of ketones
        'fr_ketone_Topliss': Descriptors.fr_ketone_Topliss(mol),    # Count of Topliss ketones
        #'fr_lactam': Descriptors.fr_lactam(mol),                    # Count of lactam groups
        #'fr_lactone': Descriptors.fr_lactone(mol),                  # Count of lactone groups
        'fr_methoxy': Descriptors.fr_methoxy(mol),                  # Count of methoxy groups
        #'fr_morpholine': Descriptors.fr_morpholine(mol),            # Count of morpholine rings
        #'fr_nitrile': Descriptors.fr_nitrile(mol),                  # Count of nitriles
        #'fr_nitro': Descriptors.fr_nitro(mol),                      # Count of nitro groups
        #'fr_nitro_arom': Descriptors.fr_nitro_arom(mol),            # Count of nitro groups on aromatic rings
        #'fr_nitro_arom_nonortho': Descriptors.fr_nitro_arom_nonortho(mol),  # Count of non-ortho nitro groups on aromatic rings
        #'fr_nitroso': Descriptors.fr_nitroso(mol),                          # Count of nitroso groups
        #'fr_oxazole': Descriptors.fr_oxazole(mol),                          # Count of oxazole rings
        'fr_oxime': Descriptors.fr_oxime(mol),                              # Count of oxime groups
        'fr_para_hydroxylation': Descriptors.fr_para_hydroxylation(mol),    # Count of para-hydroxylation sites
        'fr_phenol': Descriptors.fr_phenol(mol),                            # Count of phenols
        'fr_phenol_noOrthoHbond': Descriptors.fr_phenol_noOrthoHbond(mol),  # Count of phenols without ortho H-bond
        #'fr_phos_acid': Descriptors.fr_phos_acid(mol),                      # Count of phosphoric acid groups
        #'fr_phos_ester': Descriptors.fr_phos_ester(mol),                    # Count of phosphoric ester groups
        'fr_piperdine': Descriptors.fr_piperdine(mol),                      # Count of piperidine rings
        #'fr_piperzine': Descriptors.fr_piperzine(mol),                      # Count of piperazine rings
        #'fr_priamide': Descriptors.fr_priamide(mol),                        # Count of primary amides
        #'fr_prisulfonamd': Descriptors.fr_prisulfonamd(mol),                # Count of primary sulfonamides
        'fr_pyridine': Descriptors.fr_pyridine(mol),                        # Count of pyridine rings
        #'fr_quatN': Descriptors.fr_quatN(mol),                              # Count of quaternary nitrogens
        'fr_sulfide': Descriptors.fr_sulfide(mol),                          # Count of sulfide groups
        #'fr_sulfonamd': Descriptors.fr_sulfonamd(mol),                      # Count of sulfonamides
        #'fr_sulfone': Descriptors.fr_sulfone(mol),                          # Count of sulfone groups
        'fr_term_acetylene': Descriptors.fr_term_acetylene(mol),            # Count of terminal acetylenes
        #'fr_tetrazole': Descriptors.fr_tetrazole(mol),                      # Count of tetrazole rings
        #'fr_thiazole': Descriptors.fr_thiazole(mol),                        # Count of thiazole rings
        #'fr_thiocyan': Descriptors.fr_thiocyan(mol),                        # Count of thiocyanate groups
        #'fr_thiophene': Descriptors.fr_thiophene(mol),                      # Count of thiophene rings
        'fr_unbrch_alkane': Descriptors.fr_unbrch_alkane(mol),              # Count of unbranched alkanes
        #'fr_urea': Descriptors.fr_urea(mol),                                # Count of urea groups
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
        #'smr_VSA2' : rdMolDescriptors.SMR_VSA_(mol)[1],
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
        #'MQN7' : rdMolDescriptors.MQNs_(mol)[6],
        'MQN8' : rdMolDescriptors.MQNs_(mol)[7],
        'MQN9' : rdMolDescriptors.MQNs_(mol)[8],
        'MQN10' : rdMolDescriptors.MQNs_(mol)[9],
        'MQN11' : rdMolDescriptors.MQNs_(mol)[10],
        'MQN12' : rdMolDescriptors.MQNs_(mol)[11],
        'MQN13' : rdMolDescriptors.MQNs_(mol)[12],
        'MQN14' : rdMolDescriptors.MQNs_(mol)[13],
        #'MQN15' : rdMolDescriptors.MQNs_(mol)[14],
        'MQN16' : rdMolDescriptors.MQNs_(mol)[15],
        'MQN17' : rdMolDescriptors.MQNs_(mol)[16],
        #'MQN18' : rdMolDescriptors.MQNs_(mol)[17],
        'MQN19' : rdMolDescriptors.MQNs_(mol)[18],
        'MQN20' : rdMolDescriptors.MQNs_(mol)[19],
        'MQN21' : rdMolDescriptors.MQNs_(mol)[20],
        'MQN22' : rdMolDescriptors.MQNs_(mol)[21],
        'MQN23' : rdMolDescriptors.MQNs_(mol)[22],
        #'MQN24' : rdMolDescriptors.MQNs_(mol)[23],
        'MQN25' : rdMolDescriptors.MQNs_(mol)[24],
        'MQN26' : rdMolDescriptors.MQNs_(mol)[25],
        'MQN27' : rdMolDescriptors.MQNs_(mol)[26],
        'MQN28' : rdMolDescriptors.MQNs_(mol)[27],
        'MQN29' : rdMolDescriptors.MQNs_(mol)[28],
        'MQN30' : rdMolDescriptors.MQNs_(mol)[29],
        'MQN31' : rdMolDescriptors.MQNs_(mol)[30],
        'MQN32' : rdMolDescriptors.MQNs_(mol)[31],
        'MQN33' : rdMolDescriptors.MQNs_(mol)[32],
        #'MQN34' : rdMolDescriptors.MQNs_(mol)[33],
        'MQN35' : rdMolDescriptors.MQNs_(mol)[34],
        'MQN36' : rdMolDescriptors.MQNs_(mol)[35],
        'MQN37' : rdMolDescriptors.MQNs_(mol)[36],
        #'MQN38' : rdMolDescriptors.MQNs_(mol)[37],
        #'MQN39' : rdMolDescriptors.MQNs_(mol)[38],
        #'MQN40' : rdMolDescriptors.MQNs_(mol)[39],
        'MQN41' : rdMolDescriptors.MQNs_(mol)[40],
        'MQN42' : rdMolDescriptors.MQNs_(mol)[41]
        }
    return base_descriptors
#============================================================================================================================




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)