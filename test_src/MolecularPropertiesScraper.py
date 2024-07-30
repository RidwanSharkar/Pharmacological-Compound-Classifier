import logging
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import requests
from rdkit.Chem.rdchem import BondType
import concurrent.futures
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_smiles(pubchem_cid):
    try:
        logging.debug(f"Fetching SMILES for CID: {pubchem_cid}")
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTP Error if bad response
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(
                data['PropertyTable']['Properties']) > 0:
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            logging.debug(f"SMILES for CID {pubchem_cid}: {smiles}")
            time.sleep(0.2)  # Delay to manage API rate limits
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
        'fr_C_S': Descriptors.fr_C_S(mol),                          # thiocarbonyl
        'fr_HOCCN': Descriptors.fr_HOCCN(mol),                      # C(OH)CCN substructures
        'fr_Imine': Descriptors.fr_Imine(mol),                      # imines
        'fr_NH0': Descriptors.fr_NH0(mol),                          # tertiary amines
        'fr_NH1': Descriptors.fr_NH1(mol),                          # secondary amines
        'fr_NH2': Descriptors.fr_NH2(mol),                          # primary amines
        'fr_N_O': Descriptors.fr_N_O(mol),                          # N-O bonds
        'fr_Ndealkylation1': Descriptors.fr_Ndealkylation1(mol),    # Count of XCCNR groups
        'fr_Ndealkylation2': Descriptors.fr_Ndealkylation2(mol),    # Count of tert-alicyclic amines
        'fr_Nhpyrrole': Descriptors.fr_Nhpyrrole(mol),              # Count of H-pyrrole nitrogens
        'fr_SH': Descriptors.fr_SH(mol),                            # Count of thiol groups
        'fr_aldehyde': Descriptors.fr_aldehyde(mol),                # Count of aldehydes
        'fr_alkyl_carbamate': Descriptors.fr_alkyl_carbamate(mol),  # Count of alkyl carbamates
        'fr_alkyl_halide': Descriptors.fr_alkyl_halide(mol),        # Count of alkyl halides
        'fr_allylic_oxid': Descriptors.fr_allylic_oxid(mol),        # Count of allylic oxidation sites
        'fr_amide': Descriptors.fr_amide(mol),                      # Count of amides
        'fr_amidine': Descriptors.fr_amidine(mol),                  # Count of amidines
        'fr_aniline': Descriptors.fr_aniline(mol),                  # Count of anilines
        'fr_aryl_methyl': Descriptors.fr_aryl_methyl(mol),          # Count of aryl methyl sites
        'fr_azide': Descriptors.fr_azide(mol),                      # Count of azides
        'fr_azo': Descriptors.fr_azo(mol),                          # Count of azo groups
        # 'fr_barbitur': Descriptors.fr_barbitur(mol),                 # Potentially Redundant
        'fr_benzene': Descriptors.fr_benzene(mol),                  # Count of benzene rings
        # 'fr_benzodiazepine': Descriptors.fr_benzodiazepine(mol),     # Potentially Redundant
        'fr_bicyclic': Descriptors.fr_bicyclic(mol),                # Count of bicyclic structures
        'fr_diazo': Descriptors.fr_diazo(mol),                      # Count of diazo groups
        'fr_dihydropyridine': Descriptors.fr_dihydropyridine(mol),  # Count of dihydropyridines
        'fr_epoxide': Descriptors.fr_epoxide(mol),                  # Count of epoxide rings
        'fr_ester': Descriptors.fr_ester(mol),                      # Count of esters
        'fr_ether': Descriptors.fr_ether(mol),                      # Count of ether groups
        'fr_furan': Descriptors.fr_furan(mol),                      # Count of furan rings
        'fr_guanido': Descriptors.fr_guanido(mol),                  # Count of guanidine groups
        'fr_halogen': Descriptors.fr_halogen(mol),                  # Count of halogens
        'fr_hdrzine': Descriptors.fr_hdrzine(mol),                  # Count of hydrazine groups
        'fr_hdrzone': Descriptors.fr_hdrzone(mol),                  # Count of hydrazone groups
        'fr_imidazole': Descriptors.fr_imidazole(mol),              # Count of imidazole rings
        'fr_imide': Descriptors.fr_imide(mol),                      # Count of imide groups
        'fr_isocyan': Descriptors.fr_isocyan(mol),                  # Count of isocyanates
        'fr_isothiocyan': Descriptors.fr_isothiocyan(mol),          # Count of isothiocyanates
        'fr_ketone': Descriptors.fr_ketone(mol),                    # Count of ketones
        'fr_ketone_Topliss': Descriptors.fr_ketone_Topliss(mol),    # Count of Topliss ketones
        'fr_lactam': Descriptors.fr_lactam(mol),                    # Count of lactam groups
        'fr_lactone': Descriptors.fr_lactone(mol),                  # Count of lactone groups
        'fr_methoxy': Descriptors.fr_methoxy(mol),                  # Count of methoxy groups
        'fr_morpholine': Descriptors.fr_morpholine(mol),            # Count of morpholine rings
        'fr_nitrile': Descriptors.fr_nitrile(mol),                  # Count of nitriles
        'fr_nitro': Descriptors.fr_nitro(mol),                      # Count of nitro groups
        'fr_nitro_arom': Descriptors.fr_nitro_arom(mol),            # Count of nitro groups on aromatic rings
        'fr_nitro_arom_nonortho': Descriptors.fr_nitro_arom_nonortho(mol),  # Count of non-ortho nitro groups on aromatic rings
        'fr_nitroso': Descriptors.fr_nitroso(mol),                          # Count of nitroso groups
        'fr_oxazole': Descriptors.fr_oxazole(mol),                          # Count of oxazole rings
        'fr_oxime': Descriptors.fr_oxime(mol),                              # Count of oxime groups
        'fr_para_hydroxylation': Descriptors.fr_para_hydroxylation(mol),    # Count of para-hydroxylation sites
        'fr_phenol': Descriptors.fr_phenol(mol),                            # Count of phenols
        'fr_phenol_noOrthoHbond': Descriptors.fr_phenol_noOrthoHbond(mol),  # Count of phenols without ortho H-bond
        'fr_phos_acid': Descriptors.fr_phos_acid(mol),                      # Count of phosphoric acid groups
        'fr_phos_ester': Descriptors.fr_phos_ester(mol),                    # Count of phosphoric ester groups
        'fr_piperdine': Descriptors.fr_piperdine(mol),                      # Count of piperidine rings
        'fr_piperzine': Descriptors.fr_piperzine(mol),                      # Count of piperazine rings
        'fr_priamide': Descriptors.fr_priamide(mol),                        # Count of primary amides
        'fr_prisulfonamd': Descriptors.fr_prisulfonamd(mol),                # Count of primary sulfonamides
        'fr_pyridine': Descriptors.fr_pyridine(mol),                        # Count of pyridine rings
        'fr_quatN': Descriptors.fr_quatN(mol),                              # Count of quaternary nitrogens
        'fr_sulfide': Descriptors.fr_sulfide(mol),                          # Count of sulfide groups
        'fr_sulfonamd': Descriptors.fr_sulfonamd(mol),                      # Count of sulfonamides
        'fr_sulfone': Descriptors.fr_sulfone(mol),                          # Count of sulfone groups
        'fr_term_acetylene': Descriptors.fr_term_acetylene(mol),            # Count of terminal acetylenes
        'fr_tetrazole': Descriptors.fr_tetrazole(mol),                      # Count of tetrazole rings
        'fr_thiazole': Descriptors.fr_thiazole(mol),                        # Count of thiazole rings
        'fr_thiocyan': Descriptors.fr_thiocyan(mol),                        # Count of thiocyanate groups
        'fr_thiophene': Descriptors.fr_thiophene(mol),                      # Count of thiophene rings
        'fr_unbrch_alkane': Descriptors.fr_unbrch_alkane(mol),              # Count of unbranched alkanes
        'fr_urea': Descriptors.fr_urea(mol),                                # Count of urea groups
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
        'EState_VSA11': Descriptors.EState_VSA11(mol),                      # electrotopological state (E-state)
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
        'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),    # Number of radical electrons
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
        'SMR': Crippen.MolMR(mol),  # SMR - Molar Refractivity
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),  # LabuteASA
        'TPSA': Descriptors.TPSA(mol),  # TPSA - Topological Surface Area
        'AMW': Descriptors.MolWt(mol),  # AMW
        'ExactMW': Descriptors.ExactMolWt(mol),  # ExactMW
        'NumLipinskiHBA': Lipinski.NOCount(mol),  # to use Lipinski Rule of 5 NO
        'NumLipinskiHBD': Lipinski.NHOHCount(mol),  # to use Lipinski NHOHCount
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


def fetch_and_process(cid):
    print(f"Processing CID: {cid}")
    smiles = fetch_smiles(cid)
    if smiles:
        descriptors = compute_descriptors(smiles)
        if descriptors:
            descriptors['Cid'] = cid
            print(f"Scraped Results for CID {cid}: {descriptors}")
            return descriptors
    return None


def main():
    df = pd.read_csv('C:/Users/Lenovo/Desktop/Pharmacological-Chemical-Compound-Classifier/data/compoundCIDs_scraped.csv')
    cids = df['cid'].tolist()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_and_process, cid): cid for cid in cids}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('C:/Users/Lenovo/Desktop/Pharmacological-Chemical-Compound-Classifier/data/computedParameters_scraped.csv', index=False)
    print("Completed processing all CIDs and saved to computedParameters_scraped.csv")


if __name__ == "__main__":
    main()


