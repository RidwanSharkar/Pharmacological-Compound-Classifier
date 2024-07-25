import logging
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import requests
from rdkit.Chem.rdchem import BondType

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_smiles(pubchem_cid):
    try:
        logging.debug(f"Fetching SMILES for CID: {pubchem_cid}")
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTP Error if bad response
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(data['PropertyTable']['Properties']) > 0:
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            logging.debug(f"SMILES for CID {pubchem_cid}: {smiles}")
            time.sleep(1)  # Delay to manage API rate limits
            return smiles
        else:
            logging.error(f"No SMILES data found for CID {pubchem_cid}. Response: {data}")
            return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error for CID {pubchem_cid}: {e}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection Error for CID {pubchem_cid}: {e}")
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout Error for CID {pubchem_cid}: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception for CID {pubchem_cid}: {e}")
    except KeyError as e:
        logging.error(f"Key Error - could not find expected data for CID {pubchem_cid}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for CID {pubchem_cid}: {e}")
    return None


def compute_vsa_descriptors(mol):
    calculated_descriptors = {}

    slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
    for i, value in enumerate(slogp_vsa):
        calculated_descriptors[f'slogp_VSA{i + 1}'] = value

    smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
    for i, value in enumerate(smr_vsa):
        calculated_descriptors[f'smr_VSA{i + 1}'] = value

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


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    base_descriptors = {
        'SlogP': Crippen.MolLogP(mol),  # SlogP
        'MolarRefractivity': Crippen.MolMR(mol),  # SMR
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),  # LabuteASA
        'TopologicalPolarSurfaceArea': Descriptors.TPSA(mol),  # TPSA
        'MolecularWeight': Descriptors.MolWt(mol),  # AMW
        'ExactMW': Descriptors.ExactMolWt(mol),  # ExactMW
        'NumLipinskiHBA': Descriptors.NumHAcceptors(mol),  # NumLipinskiHBA
        'NumLipinskiHBD': Descriptors.NumHDonors(mol),  # NumLipinskiHBD
        'RotatableBondCount': Descriptors.NumRotatableBonds(mol),  # NumRotatableBonds
        'HBondDonorCount': Descriptors.NumHDonors(mol),  # NumHBD
        'HBondAcceptorCount': Descriptors.NumHAcceptors(mol),  # NumHBA
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


def main():
    # Load data
    df = pd.read_csv('C:/Users/Lenovo/Desktop/Psychoactive-Compounds-Analysis/data/compoundCIDs_scraped.csv')

    # Prepare to collect data
    results = []

    for cid in df['cid'][:30]: # to test
        logging.debug(f"Processing CID: {cid}")
        print(f"Processing CID: {cid}")
        smiles = fetch_smiles(cid)
        if smiles:
            descriptors = compute_descriptors(smiles)
            if descriptors:
                descriptors['Cid'] = cid
                results.append(descriptors)
                print(f"Scraped Results for CID {cid}: {descriptors}")  # Print the whole row of scraped results

    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    cols = ['Cid'] + [col for col in results_df.columns if col != 'Cid']
    results_df = results_df[cols]
    print(results_df)

    # Save the DataFrame to a CSV file
    results_df.to_csv('C:/Users/Lenovo/Desktop/Psychoactive-Compounds-Analysis/data/computed_descriptors.csv', index=False)
    print("Completed processing all CIDs and saved to computed_descriptors.csv")


if __name__ == "__main__":
    main()