from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import requests


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


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    base_descriptors = {
        'MolecularWeight': Descriptors.MolWt(mol),
        'HBondDonorCount': Descriptors.NumHDonors(mol),
        'HBondAcceptorCount': Descriptors.NumHAcceptors(mol),
        'RotatableBondCount': Descriptors.NumRotatableBonds(mol),
        'TopologicalPolarSurfaceArea': Descriptors.TPSA(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
        'NumAtoms': mol.GetNumAtoms(),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'HallKierAlpha': rdMolDescriptors.CalcHallKierAlpha(mol),
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
        'SlogP': Crippen.MolLogP(mol),
        'ExactMW': Descriptors.ExactMolWt(mol),
        'NumRings': rdMolDescriptors.CalcNumRings(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        # Chi indices
        'Chi0v': rdMolDescriptors.CalcChi0v(mol),
        'Chi1v': rdMolDescriptors.CalcChi1v(mol),
        'Chi2v': rdMolDescriptors.CalcChi2v(mol),
        'Chi3v': rdMolDescriptors.CalcChi3v(mol),
        'Chi4v': rdMolDescriptors.CalcChi4v(mol),
        'Chi1n': rdMolDescriptors.CalcChi1n(mol),
        'Chi2n': rdMolDescriptors.CalcChi2n(mol),
        'Chi3n': rdMolDescriptors.CalcChi3n(mol),
        'Chi4n': rdMolDescriptors.CalcChi4n(mol),
        'kappa1': rdMolDescriptors.CalcKappa1(mol),
        'kappa2': rdMolDescriptors.CalcKappa2(mol),
        'kappa3': rdMolDescriptors.CalcKappa3(mol),
    }

    base_descriptors.update(compute_vsa_descriptors(mol))

    return base_descriptors


# Example use:
smiles = fetch_smiles('10836')
descriptors = compute_descriptors(smiles)
print(descriptors)
