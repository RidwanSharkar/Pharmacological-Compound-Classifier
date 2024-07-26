import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast  # safe evaluation of string literals to lists
import joblib

# DATA PROCESSING & FEATURE SELECTION===================================================================================
dataset = 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\computedParameters_scraped.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\compoundClassifications_scraped.csv'
data = pd.read_csv(dataset)
activities = pd.read_csv(pharmacologicalActivities)

merged_data = pd.merge(data, activities, on='CID', how='left')
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# Transform 'Activities' into binary matrix (Multi-label - Binary Relevance Model with Random Forest ** HAMMING LOSS **
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

features = merged_data[['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].fillna(0)

# TRAIN MODEL=============================================================================
X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)

# Save Results
joblib.dump(model, 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\model\\RandomForestModel.pkl')
joblib.dump(mlb, 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\model\\MultiLabelBinarizer.pkl')

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
test_data = data[data['CID'] == 44558536]

if not test_data.empty:
    features = test_data[['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].fillna(0)
    predicted_activities_encoded = model.predict(features)
    predicted_activities_labels = mlb.inverse_transform(predicted_activities_encoded)
    print("Predicted Activities for CID 44558536:", predicted_activities_labels)

# FEATURE IMPORTANCES=====================================================================
feature_names = merged_data[['SlogP', 'SMR', 'LabuteASA', 'TPSA', 'AMW', 'ExactMW', 'NumLipinskiHBA', 'NumLipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAmideBonds', 'NumHeteroAtoms', 'NumHeavyAtoms', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles', 'NumAliphaticCarbocycles', 'FractionCSP3', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n', 'HallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'slogp_VSA1', 'slogp_VSA2', 'slogp_VSA3', 'slogp_VSA4', 'slogp_VSA5', 'slogp_VSA6', 'slogp_VSA7', 'slogp_VSA8', 'slogp_VSA9', 'slogp_VSA10', 'slogp_VSA11', 'slogp_VSA12', 'smr_VSA1', 'smr_VSA2', 'smr_VSA3', 'smr_VSA4', 'smr_VSA5', 'smr_VSA6', 'smr_VSA7', 'smr_VSA8', 'smr_VSA9', 'smr_VSA10', 'peoe_VSA1', 'peoe_VSA2', 'peoe_VSA3', 'peoe_VSA4', 'peoe_VSA5', 'peoe_VSA6', 'peoe_VSA7', 'peoe_VSA8', 'peoe_VSA9', 'peoe_VSA10', 'peoe_VSA11', 'peoe_VSA12', 'peoe_VSA13', 'peoe_VSA14', 'MQN1', 'MQN2', 'MQN3', 'MQN4', 'MQN5', 'MQN6', 'MQN7', 'MQN8', 'MQN9', 'MQN10', 'MQN11', 'MQN12', 'MQN13', 'MQN14', 'MQN15', 'MQN16', 'MQN17', 'MQN18', 'MQN19', 'MQN20', 'MQN21', 'MQN22', 'MQN23', 'MQN24', 'MQN25', 'MQN26', 'MQN27', 'MQN28', 'MQN29', 'MQN30', 'MQN31', 'MQN32', 'MQN33', 'MQN34', 'MQN35', 'MQN36', 'MQN37', 'MQN38', 'MQN39', 'MQN40', 'MQN41', 'MQN42']].columns
importances = model.feature_importances_

importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
for index, row in importances_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")
top_n = 120
print("\nTop Important Features:")
top_features = importances_df.head(top_n)
for index, row in top_features.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")