# model\Model-RandomForest.py:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast 
import joblib

# DATA PROCESSING & FEATURE SELECTION===================================================================================
# dataset = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\original\\psychoactiveCompounds_Dataset.csv'
dataset = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\computedParameters_scraped.csv'
# pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\original\\compoundInfo_scraped.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv'
data = pd.read_csv(dataset)
activities = pd.read_csv(pharmacologicalActivities)

merged_data = pd.merge(data, activities, on='CID', how='left')
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# Transform 'Activities' into binary matrix (Multi-label - Binary Relevance Model with Random Forest ** HAMMING LOSS **
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

features = merged_data[[
    #'fr_Al_COO',
    #'fr_Al_OH',
    #'fr_Al_OH_noTert',
    #'fr_ArN',
    #'fr_Ar_N',
    #'fr_Ar_NH',
    #'fr_Ar_OH',
    #'fr_COO',
    #'fr_COO2',
    #'fr_C_O',
    #'fr_C_O_noCOO',
    #'fr_C_S',
    #'fr_HOCCN',
    #'fr_Imine',
    #'fr_NH0',
    #'fr_NH1',
    #'fr_NH2',
    #'fr_N_O',
    #'fr_Ndealkylation1',
    #'fr_Ndealkylation2',
    #'fr_Nhpyrrole',
    #'fr_SH',
    #'fr_aldehyde',
    #'fr_alkyl_carbamate',
    #'fr_alkyl_halide',
    #'fr_allylic_oxid',
    #'fr_amide',
    #'fr_amidine',
    #'fr_aniline',
    #'fr_aryl_methyl',
    #'fr_azide',
    #'fr_azo',
    #'fr_benzene',
    #'fr_bicyclic',
    #'fr_diazo',
    #'fr_dihydropyridine',
    #'fr_epoxide',
    #'fr_ester',
    #'fr_ether',
    #'fr_furan',
    #'fr_guanido',
    #'fr_halogen',
    #'fr_hdrzine',
    #'fr_hdrzone',
    #'fr_imidazole',
    #'fr_imide',
    #'fr_isocyan',
    #'fr_isothiocyan',
    #'fr_ketone',
    #'fr_ketone_Topliss',
    #'fr_lactam',
    #'fr_lactone',
    #'fr_methoxy',
    #'fr_morpholine',
    #'fr_nitrile',
    #'fr_nitro',
    #'fr_nitro_arom',
    #'fr_nitro_arom_nonortho',
    #'fr_nitroso',
    #'fr_oxazole',
    #'fr_oxime',
    #'fr_para_hydroxylation',
    #'fr_phenol',
    #'fr_phenol_noOrthoHbond',
    #'fr_phos_acid',
    #'fr_phos_ester',
    #'fr_piperdine',
    #'fr_piperzine',
    #'fr_priamide',
    #'fr_prisulfonamd',
    #'fr_pyridine',
    #'fr_quatN',
    #'fr_sulfide',
    #'fr_sulfonamd',
    #'fr_sulfone',
    #'fr_term_acetylene',
    #'fr_tetrazole',
    #'fr_thiazole',
    #'fr_thiocyan',
    #'fr_thiophene',
    #'fr_unbrch_alkane',
    #'fr_urea',
    #'EState_VSA1',
    #'EState_VSA2',
    #'EState_VSA3',
    #'EState_VSA4',
    #'EState_VSA5',
    #'EState_VSA6',
    #'EState_VSA7',
    #'EState_VSA8',
    #'EState_VSA9',
    #'EState_VSA10',
    #'EState_VSA11',
    #'VSA_EState1',
    #'VSA_EState2',
    #'VSA_EState3',
    #'VSA_EState4',
    #'VSA_EState5',
    #'VSA_EState6',
    #'VSA_EState7',
    #'VSA_EState8',
    #'VSA_EState9',
    #'VSA_EState10',
    #'MaxAbsEStateIndex',
    #'MaxEStateIndex',
    #'MinAbsEStateIndex',
    #'MinEStateIndex',
    #'qed',
    #'HeavyAtomMolWt',
    #'NumValenceElectrons',
    #'NumRadicalElectrons',
    #'MaxPartialCharge',
    #'MinPartialCharge',
    #'MaxAbsPartialCharge',
    #'MinAbsPartialCharge',
    #'FpDensityMorgan1',
    #'FpDensityMorgan2',
    #'FpDensityMorgan3',
    #'BCUT2D_MWHI',
    #'BCUT2D_MWLOW',
    #'BCUT2D_CHGHI',
    #'BCUT2D_CHGLO',
    #'BCUT2D_LOGPHI',
    #'BCUT2D_LOGPLOW',
    #'BCUT2D_MRHI',
    #'BCUT2D_MRLOW',
    #'AvgIpc',
    #'BalabanJ',
    #'BertzCT',
'SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].fillna(0)

# TRAIN MODEL=============================================================================
X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)

# Save Results
joblib.dump(model, 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\RandomForestModel.pkl')
joblib.dump(mlb, 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\MultiLabelBinarizer.pkl')

# EVALUATE================================================================================
accuracies = []
for i, label in enumerate(mlb.classes_):
    accuracy = accuracy_score(y_test[:, i], y_prediction[:, i])
    accuracies.append(accuracy)
    print(f"Accuracy for {label}: {accuracy:.2f}")

# AVERAGE ACCURACY========================================================================
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy: {average_accuracy:.2f}")

# EXAMPLE TEST============================================================================
# test_data = data[data['CID'] == 44558536]
# if not test_data.empty:
#    features = test_data[['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].fillna(0)
#    predicted_activities_encoded = model.predict(features)
#    predicted_activities_labels = mlb.inverse_transform(predicted_activities_encoded)
#    print("Predicted Activities for CID 44558536:", predicted_activities_labels)

# FEATURE IMPORTANCES=====================================================================
feature_names = merged_data[[
    #'fr_Al_COO',
    #'fr_Al_OH',
    #'fr_Al_OH_noTert',
    #'fr_ArN',
    #'fr_Ar_N',
    #'fr_Ar_NH',
    #'fr_Ar_OH',
    #'fr_COO',
    #'fr_COO2',
    #'fr_C_O',
    #'fr_C_O_noCOO',
    #'fr_C_S',
    #'fr_HOCCN',
    #'fr_Imine',
    #'fr_NH0',
    #'fr_NH1',
    #'fr_NH2',
    #'fr_N_O',
    #'fr_Ndealkylation1',
    #'fr_Ndealkylation2',
    #'fr_Nhpyrrole',
    #'fr_SH',
    #'fr_aldehyde',
    #'fr_alkyl_carbamate',
    #'fr_alkyl_halide',
    #'fr_allylic_oxid',
    #'fr_amide',
    #'fr_amidine',
    #'fr_aniline',
    #'fr_aryl_methyl',
    #'fr_azide',
    #'fr_azo',
    #'fr_benzene',
    #'fr_bicyclic',
    #'fr_diazo',
    #'fr_dihydropyridine',
    #'fr_epoxide',
    #'fr_ester',
    #'fr_ether',
    #'fr_furan',
    #'fr_guanido',
    #'fr_halogen',
    #'fr_hdrzine',
    #'fr_hdrzone',
    #'fr_imidazole',
    #'fr_imide',
    #'fr_isocyan',
    #'fr_isothiocyan',
    #'fr_ketone',
    #'fr_ketone_Topliss',
    #'fr_lactam',
    #'fr_lactone',
    #'fr_methoxy',
    #'fr_morpholine',
    #'fr_nitrile',
    #'fr_nitro',
    #'fr_nitro_arom',
    #'fr_nitro_arom_nonortho',
    #'fr_nitroso',
    #'fr_oxazole',
    #'fr_oxime',
    #'fr_para_hydroxylation',
    #'fr_phenol',
    #'fr_phenol_noOrthoHbond',
    #'fr_phos_acid',
    #'fr_phos_ester',
    #'fr_piperdine',
    #'fr_piperzine',
    #'fr_priamide',
    #'fr_prisulfonamd',
    #'fr_pyridine',
    #'fr_quatN',
    #'fr_sulfide',
    #'fr_sulfonamd',
    #'fr_sulfone',
    #'fr_term_acetylene',
    #'fr_tetrazole',
    #'fr_thiazole',
    #'fr_thiocyan',
    #'fr_thiophene',
    #'fr_unbrch_alkane',
    #'fr_urea',
    #'EState_VSA1',
    #'EState_VSA2',
    #'EState_VSA3',
    #'EState_VSA4',
    #'EState_VSA5',
    #'EState_VSA6',
    #'EState_VSA7',
    #'EState_VSA8',
    #'EState_VSA9',
    #'EState_VSA10',
    #'EState_VSA11',
    #'VSA_EState1',
    #'VSA_EState2',
    #'VSA_EState3',
    #'VSA_EState4',
    #'VSA_EState5',
    #'VSA_EState6',
    #'VSA_EState7',
    #'VSA_EState8',
    #'VSA_EState9',
    #'VSA_EState10',
    #'MaxAbsEStateIndex',
    #'MaxEStateIndex',
    #'MinAbsEStateIndex',
    #'MinEStateIndex',
    #'qed',
    #'HeavyAtomMolWt',
    #'NumValenceElectrons',
    #'NumRadicalElectrons',
    #'MaxPartialCharge',
    #'MinPartialCharge',
    #'MaxAbsPartialCharge',
    #'MinAbsPartialCharge',
    #'FpDensityMorgan1',
    #'FpDensityMorgan2',
    #'FpDensityMorgan3',
    #'BCUT2D_MWHI',
    #'BCUT2D_MWLOW',
    #'BCUT2D_CHGHI',
    #'BCUT2D_CHGLO',
    #'BCUT2D_LOGPHI',
    #'BCUT2D_LOGPLOW',
    #'BCUT2D_MRHI',
    #'BCUT2D_MRLOW',
    #'AvgIpc',
    #'BalabanJ',
    #'BertzCT',
'SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].columns
importances = model.feature_importances_

importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
for index, row in importances_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")



