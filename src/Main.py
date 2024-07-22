import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import joblib  # For loading the model
import requests
from rdkit.Chem.rdchem import BondType

# LOAD
model = joblib.load('C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\model\\RandomForestModel.pkl')
mlb = joblib.load('C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\model\\MultiLabelBinarizer.pkl')
# all_descriptors = [desc[0] for desc in Descriptors.descList]
# print(all_descriptors)


def fetch_smiles(pubchem_cid):
    """ Fetch the SMILES string from PubChem using its REST API. """
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
    response = requests.get(url)
    data = response.json()
    return data['PropertyTable']['Properties'][0]['CanonicalSMILES']


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

# def count_amide_bonds(mol):
#    """Count the number of amide bonds in a molecule."""
#    amide_bond_count = 0
#    for bond in mol.GetBonds():
#        if bond.GetBondType() == BondType.AMIDE:
#            amide_bond_count += 1
#    return amide_bond_count

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
        'NumLipinskiHBA': Descriptors.NumHAcceptors(mol),  # NumLipinskiHBA
        'NumLipinskiHBD': Descriptors.NumHDonors(mol),  # NumLipinskiHBD
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # NumRotatableBonds
        'NumHBD': Descriptors.NumHDonors(mol),  # NumHBD
        'NumHBA': Descriptors.NumHAcceptors(mol),  # NumHBA
        'NumAmideBonds': count_amide_bonds(mol),
        'NumHeteroAtoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),  # NumHeavyAtoms
        'NumAtoms': mol.GetNumAtoms(),  # NumAtoms
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
        'Chi4v': rdMolDescriptors.CalcChi4v(mol),  # Chi x9
        'Chi1n': rdMolDescriptors.CalcChi1n(mol),
        'Chi2n': rdMolDescriptors.CalcChi2n(mol),
        'Chi3n': rdMolDescriptors.CalcChi3n(mol),
        'Chi4n': rdMolDescriptors.CalcChi4n(mol),
        'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(mol),  # HallKierAlpha
        'kappa1': rdMolDescriptors.CalcKappa1(mol),  # kappa1
        'kappa2': rdMolDescriptors.CalcKappa2(mol),  # kappa2
        'kappa3': rdMolDescriptors.CalcKappa3(mol),  # kappa3 = 39
    }
    base_descriptors.update(compute_vsa_descriptors(mol))
    base_descriptors.update(compute_mqn_descriptors(mol))
    return base_descriptors


def predict_activities(descriptors):
    df = pd.DataFrame([descriptors])
    expected_feature_order = ['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']
    prediction_features = df[expected_feature_order]
    prediction = model.predict(prediction_features)
    predicted_labels = mlb.inverse_transform(prediction)
    return predicted_labels


def main():
    cid = input("Enter CID: ")
    smiles = fetch_smiles(cid)
    descriptors = compute_descriptors(smiles)
    if descriptors:
        activities = predict_activities(descriptors)
        print(f"Predicted Activities for CID {cid}: {activities}")
    else:
        print("Error: Invalid SMILES or CID not found.")


if __name__ == "__main__":
    main()
