# model\DNN.py:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast 
import shap
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
import optuna

# Load dataset
dataset = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\computedParameters_scraped.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv'
data = pd.read_csv(dataset)
activities = pd.read_csv(pharmacologicalActivities)

merged_data = pd.merge(data, activities, on='CID', how='left')
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# Transform 'Activities' into binary matrix
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

# Select and scale features
features = merged_data[[
    #'fr_Al_COO',
    'fr_Al_OH',            # 94
    #'fr_Al_OH_noTert',     # 109
    #'fr_ArN',
    #'fr_Ar_N',
    #'fr_Ar_NH',
    #'fr_Ar_OH',
    #'fr_COO',
    #'fr_COO2',
    'fr_C_O',                      # 115
    #'fr_C_O_noCOO',
    #'fr_C_S',
    #'fr_HOCCN',
    #'fr_Imine',
    'fr_NH0',                   # 98
    #'fr_NH1',                  # 114 
    #'fr_NH2',
    #'fr_N_O',
    #'fr_Ndealkylation1',       # 111
    #'fr_Ndealkylation2',
    #'fr_Nhpyrrole',
    #'fr_SH',
    #'fr_aldehyde',
    #'fr_alkyl_carbamate',
    #'fr_alkyl_halide',
    #'fr_allylic_oxid',             # 97
    #'fr_amide',
    #'fr_amidine',
    #'fr_aniline',
    #'fr_aryl_methyl',
    #'fr_azide',
    #'fr_azo',
    #'fr_benzene',                  # 116
    'fr_bicyclic',                 # 90
    #'fr_diazo',
    #'fr_dihydropyridine',
    #'fr_epoxide',
    #'fr_ester',                    
    'fr_ether',                    # 102
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
    # 'EState_VSA11',
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
    # 'NumRadicalElectrons',
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
    # 'AMW',
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
    # 'NumAromaticCarbocycles',
    # 'NumSaturatedCarbocycles',
    # 'NumAliphaticCarbocycles',
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
    # 'slogp_VSA7',
    'slogp_VSA8',           # 103 
    # 'slogp_VSA9',
    'slogp_VSA10',              # 112
    'slogp_VSA11',
    'slogp_VSA12',
    'smr_VSA1',             # 81
    # 'smr_VSA2',
    'smr_VSA3',             # 84
    'smr_VSA4',             # 80
    'smr_VSA5',             # 33
    'smr_VSA6',             # 43 
    'smr_VSA7',             # 45
    # 'smr_VSA8',
    'smr_VSA9',             # 106
    'smr_VSA10',                    # 65
    'peoe_VSA1',                    # 63
    'peoe_VSA2',                    # 85
    'peoe_VSA3',                    # 101
    # 'peoe_VSA4',
    # 'peoe_VSA5',
    'peoe_VSA6',                    # 60
    'peoe_VSA7',                    # 23
    'peoe_VSA8',                    # 42
    'peoe_VSA9',                    # 50
    'peoe_VSA10',                   # 64
    'peoe_VSA11',                   # 86
    'peoe_VSA12',                   # 105 
    # 'peoe_VSA13',
    'peoe_VSA14',                   # 111
    #'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6',
    #'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12',
    'MQN13', 
    #'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18',
    #'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24',
    #'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 
    'MQN30', # 70
    'MQN31', # 62
    #'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36',
    #'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42'
    ]].fillna(0)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)


# OPTUNA HYPERPARAMETER TRIALS===================================================================================

def objective(trial):
    # Define the hyperparameters to be tuned
    n_layers = trial.suggest_int('n_layers', 2, 5)
    neurons = trial.suggest_categorical('neurons', [256, 512, 1024, 2048])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = Sequential()
    input_shape = X_train.shape[1]
    
    # First layer
    model.add(Dense(2048, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Additional layers based on the number of layers
    for _ in range(n_layers):
        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(mlb.classes_.shape[0], activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.20, callbacks=[early_stopping, reduce_lr], verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return accuracy

# OPTUNA HYPERPARAMETER TRIALS===================================================================================


# Create the study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters: ', study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params

model = Sequential()
input_shape = X_train.shape[1]
model.add(Dense(2048, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(best_params['l2_reg'])))
model.add(BatchNormalization())
model.add(Dropout(best_params['dropout_rate']))

for _ in range(best_params['n_layers']):
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Dropout(best_params['dropout_rate']))

model.add(Dense(mlb.classes_.shape[0], activation='sigmoid'))

optimizer = Adam(learning_rate=best_params['learning_rate'], clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

model.fit(X_train, y_train, epochs=100, batch_size=best_params['batch_size'], validation_split=0.15, callbacks=[early_stopping, reduce_lr])

# Evaluate the final model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

model.save('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\DNNModel.h5')
joblib.dump(mlb, 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\MultiLabelBinarizer.pkl')



#========================================================================================================================================================================================================================================================================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast 
import shap
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
import optuna

# Load dataset
dataset = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\computedParameters_scraped.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv'
data = pd.read_csv(dataset)
activities = pd.read_csv(pharmacologicalActivities)

merged_data = pd.merge(data, activities, on='CID', how='left')
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# Transform 'Activities' into binary matrix
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

# Select and scale features
features = merged_data[[
    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_N', 'fr_Ar_NH',
    'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
    'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
    'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_benzene',
    'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
    'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
    'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
    'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
    'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
    'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
    'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
    'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'EState_VSA11',
    'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9', 'VSA_EState10', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex',
    'MinEStateIndex', 'qed', 'HeavyAtomMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'SlogP', 'SMR', 'LabuteASA',
    'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA',
    'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings',
    'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles',
    'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles',
    'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
    'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4',
    'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11',
    'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7',
    'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5',
    'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13',
    'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11',
    'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23',
    'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35',
    'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].fillna(0)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)

def objective(trial):
    print("Starting a new trial...")
    n_layers = trial.suggest_int('n_layers', 2, 6)
    neurons = trial.suggest_categorical('neurons', [256, 512, 1024, 2048])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(n_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(mlb.classes_.shape[0], activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.15, callbacks=[early_stopping, reduce_lr], verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Trial completed: Loss = {loss}, Accuracy = {accuracy}")
    return accuracy

print("Creating study and starting optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

print('Best hyperparameters: ', study.best_params)