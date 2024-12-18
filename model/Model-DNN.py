# model\Model-DeepNeuralNetwork.py:
#    tensorflow.python.keras.layers - import bug
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.metrics import AUC, Precision, Recall # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast 
import joblib
import tensorflow as tf


# DATA PROCESSING & FEATURES + IMPORTANCES =======================================================================================
dataset = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\computedParameters_scraped.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv'

data = pd.read_csv(dataset)
activities_unfiltered = pd.read_csv(pharmacologicalActivities)
psychoactiveCIDs = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\psychoactiveCIDs.csv')

activities = activities_unfiltered[activities_unfiltered['CID'].isin(psychoactiveCIDs['CID'])]

merged_data = pd.merge(data, activities, on='CID', how='left')
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# Transform 'Activities' column into binary matrix (Multi-label) - ** HAMMING LOSS **
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

features = merged_data[[
    'fr_Al_COO',
    'fr_Al_OH',            # 94
    'fr_Al_OH_noTert',     # 109
    'fr_ArN',
    'fr_Ar_N',
    'fr_Ar_NH',
    'fr_Ar_OH',
    'fr_COO',
    'fr_COO2',
    'fr_C_O',                      # 115
    'fr_C_O_noCOO',
    #'fr_C_S',
    'fr_HOCCN',
    'fr_Imine',
    'fr_NH0',                   # 98
    'fr_NH1',                  # 114 
    'fr_NH2',
    #'fr_N_O',
    'fr_Ndealkylation1',       # 111
    'fr_Ndealkylation2',
    'fr_Nhpyrrole',
    #'fr_SH',
    #'fr_aldehyde',
    #'fr_alkyl_carbamate',
    #'fr_alkyl_halide',
    'fr_allylic_oxid',             # 97
    'fr_amide',
    #'fr_amidine',
    'fr_aniline',
    'fr_aryl_methyl',
    #'fr_azide',
    #'fr_azo',
    'fr_benzene',                  # 116
    'fr_bicyclic',                 # 90
    #'fr_diazo',
    #'fr_dihydropyridine',
    'fr_epoxide',
    'fr_ester',                    
    'fr_ether',                    # 102
    #'fr_furan',
    'fr_guanido',
    'fr_halogen',
    #'fr_hdrzine',
    #'fr_hdrzone',
    #'fr_imidazole',
    #'fr_imide',
    #'fr_isocyan',
    #'fr_isothiocyan',
    'fr_ketone',
    'fr_ketone_Topliss',
    #'fr_lactam',
    #'fr_lactone',
    'fr_methoxy',
    #'fr_morpholine',
    #'fr_nitrile',
    #'fr_nitro',
    #'fr_nitro_arom',
    #'fr_nitro_arom_nonortho',
    #'fr_nitroso',
    #'fr_oxazole',
    'fr_oxime',
    'fr_para_hydroxylation',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    #'fr_phos_acid',
    #'fr_phos_ester',
    'fr_piperdine',
    #'fr_piperzine',
    #'fr_priamide',
    #'fr_prisulfonamd',
    'fr_pyridine',
    #'fr_quatN',
    'fr_sulfide',
    #'fr_sulfonamd',
    #'fr_sulfone',
    'fr_term_acetylene',
    #'fr_tetrazole',
    #'fr_thiazole',
    #'fr_thiocyan',
    #'fr_thiophene',
    'fr_unbrch_alkane',
    #'fr_urea',
    'EState_VSA1',              # 53   
    'EState_VSA2',              # 77
    'EState_VSA3',              # 58
    'EState_VSA4',              # 55
    'EState_VSA5',              # 68
    'EState_VSA6',              # 83 
    'EState_VSA7',              # 72
    'EState_VSA8',              # 37
    'EState_VSA9',              # 71
    'EState_VSA10',             # 78
    #'EState_VSA11',
    'VSA_EState1',              # 67
    'VSA_EState2',              # 15
    'VSA_EState3',              # 17
    'VSA_EState4',              # 11
    'VSA_EState5',              # 4
    'VSA_EState6',              # 6
    'VSA_EState7',              # 12
    'VSA_EState8',              # 9
    'VSA_EState9',              # 29
    'VSA_EState10',             # 92
    'MaxAbsEStateIndex',        # 7
    'MaxEStateIndex',           # 5
    'MinAbsEStateIndex',        # 82
    'MinEStateIndex',           # 31
    'qed',                      # 20
    'HeavyAtomMolWt',           # 74
    'NumValenceElectrons',      # 79    
    #'NumRadicalElectrons',
    'MaxPartialCharge',         # 34       
    'MinPartialCharge',         # 47
    'MaxAbsPartialCharge',      # 54       
    'MinAbsPartialCharge',      # 25
    'FpDensityMorgan1',         # 18
    'FpDensityMorgan2',         # 19
    'FpDensityMorgan3',         # 16
    'BCUT2D_MWHI',                  # 59
    'BCUT2D_MWLOW',             # 13
    'BCUT2D_CHGHI',             # 3
    'BCUT2D_CHGLO',             # 1
    'BCUT2D_LOGPHI',            # 8
    'BCUT2D_LOGPLOW',           # 2
    'BCUT2D_MRHI',              # 22
    'BCUT2D_MRLOW',             # 10 
    'AvgIpc',                   # 14
    'BalabanJ',                 # 56
    'BertzCT',                  # 21
    'SlogP',                    # 41
    'SMR',                      # 63
    'LabuteASA',                # 69
    'TPSA',                     # 32
    #'AMW',
    'ExactMW',                  # 66
    'NumLipinskiHBA',           # 88
    'NumLipinskiHBD',           # 99
    'NumRotatableBonds',        # 76
    'NumHBD',                   # 96
    'NumHBA',                   # 91
    'NumAmideBonds',
    'NumHeteroAtoms',           # 93 
    'NumHeavyAtoms',            # 89
    'NumAtoms',                 # 87
    'NumRings',                 # 100
    'NumAromaticRings',             # 117
    'NumSaturatedRings',            # 110
    'NumAliphaticRings',            # 107
    'NumAromaticHeterocycles',
    'NumSaturatedHeterocycles',     # 104
    'NumAliphaticHeterocycles',     # 95
    'NumAromaticCarbocycles',
    'NumSaturatedCarbocycles',
    'NumAliphaticCarbocycles',
    'FractionCSP3',         # 30
    'Chi0v',                # 75
    'Chi1v',                # 44
    'Chi2v',                # 40
    'Chi3v',                # 36
    'Chi4v',                # 24
    'Chi1n',                # 39
    'Chi2n',                # 35
    'Chi3n',                # 27
    'Chi4n',                # 26
    'HallKierAlpha',        # 51
    'kappa1',               # 73
    'kappa2',               # 57
    'kappa3',               # 38
    'slogp_VSA1',               # 83
    'slogp_VSA2',           # 28
    'slogp_VSA3',               # 61
    'slogp_VSA4',               # 83
    'slogp_VSA5',           # 52
    'slogp_VSA6',           # 49
    'slogp_VSA7',
    'slogp_VSA8',           # 103 
    #'slogp_VSA9',
    'slogp_VSA10',              # 112
    'slogp_VSA11',
    'slogp_VSA12',
    'smr_VSA1',             # 81
    #'smr_VSA2',
    'smr_VSA3',             # 84
    'smr_VSA4',             # 80
    'smr_VSA5',             # 33
    'smr_VSA6',             # 43 
    'smr_VSA7',             # 45
    #'smr_VSA8',
    'smr_VSA9',             # 106
    'smr_VSA10',                    # 65
    'peoe_VSA1',                    # 63
    'peoe_VSA2',                    # 85
    'peoe_VSA3',                    # 101
    'peoe_VSA4',
    'peoe_VSA5',
    'peoe_VSA6',                    # 60
    'peoe_VSA7',                    # 23
    'peoe_VSA8',                    # 42
    'peoe_VSA9',                    # 50
    'peoe_VSA10',                   # 64
    'peoe_VSA11',                   # 86
    'peoe_VSA12',                   # 105 
    'peoe_VSA13',
    'peoe_VSA14',                   # 111
    'MQN1', 'MQN2', 'MQN3', # 'MQN4', # 'MQN5', 
    'MQN6', # 'MQN7', 
    'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12',
    'MQN13', 'MQN14', # 'MQN15', 
    'MQN16', 'MQN17', # 'MQN18',
    'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', # 'MQN24',
    'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 
    'MQN30', # 70
    'MQN31', # 62
    'MQN32', 'MQN33', # 'MQN34', 
    'MQN35', 'MQN36',
    'MQN37', #'MQN38', # 'MQN39', # 'MQN40', 
    'MQN41', 'MQN42'
]].fillna(0)




# TRAIN MODEL=====================================================================================================================
scaler = StandardScaler()
features = scaler.fit_transform(features)                                                                          # NORMALIZATION 
X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(1240, activation='relu', input_shape=(X_train.shape[1],),),
    BatchNormalization(),
    Dropout(0.20), 
    Dense(1024, activation='relu',), # TRY L2
    BatchNormalization(), 
    Dropout(0.20),
    Dense(1024, activation='relu',), # TRY L2
    BatchNormalization(), 
    Dropout(0.30),
    #Dense(768, activation='relu',),
    #BatchNormalization(),
    #Dropout(0.20),
    Dense(mlb.classes_.shape[0], activation='sigmoid')])  # Output layer with 1 neuron/class                   # SIGMOID ACTIVATION
adam_optimizer = Adam(learning_rate=0.00035, clipnorm=1.0)                                                      # GRADIENT CLIPPING
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])       
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)                # LEARNING RATE SCHEDULING 
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  , callbacks=[early_stopping] # EARLY STOPPING 
model.fit(X_train, y_train, epochs=70, batch_size=24, validation_split = 0.20)  


# OG MODEL========================================================================================================================
# model = Sequential([
#    Dense(1536, activation='relu', input_shape=(X_train.shape[1],),),
#    BatchNormalization(),
#    Dropout(0.30), 
#    Dense(1024, activation='relu',),
#    BatchNormalization(),
#    Dropout(0.35),
#    Dense(256, activation='relu',),
#    BatchNormalization(),
#    Dropout(0.35),
#    Dense(mlb.classes_.shape[0], activation='sigmoid')])               # Output layer with 1 neuron/class       SIGMOID ACTIVATION
#adam_optimizer = Adam(learning_rate=0.0004, clipnorm=1.0)                                                      # GRADIENT CLIPPING
#model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])       
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)   # (OnPlateau) LEARNING RATE SCHEDULING 
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)                        # EARLY STOPPING 
#model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split = 0.20, callbacks=[early_stopping])          

# ACCURACY========================================================================================================================
metrics = model.evaluate(X_test, y_test)
loss = metrics[0]
accuracy = metrics[1]
auc = metrics[2]
precision = metrics[3]
recall = metrics[4]

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
print(f"Test AUC: {auc}")
print(f"Test precision: {precision}")
print(f"Test recall: {recall}")

model.save('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\DNNModel.h5')
joblib.dump(mlb, 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\MultiLabelBinarizer.pkl')
