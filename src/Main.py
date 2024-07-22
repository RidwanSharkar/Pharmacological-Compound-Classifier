import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ast  # Safe evaluation of string literals to lists

# Load datasets
dataset = 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\psychoactive compounds.csv'
pharmacologicalActivities = 'C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\compound_info.csv'

data = pd.read_csv(dataset)
activities = pd.read_csv(pharmacologicalActivities)

# Assuming 'CID' is a common column to merge on, or use index if they are aligned
merged_data = pd.merge(data, activities, on='CID', how='left')

# Handle missing activities and convert string representations of lists to lists
merged_data['Activities'] = merged_data['Activities'].fillna('[]').apply(ast.literal_eval)

# Filter out entries without activities
merged_data = merged_data[merged_data['Activities'].map(len) > 0]

# MultiLabel Binarizer to transform the 'Activities' column into a binary matrix
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(merged_data['Activities'])

# Ensure the molecular properties columns exist and handle missing values if necessary
features = merged_data[['NumHBD', 'NumHBA', 'ExactMW', 'AMW']].fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier for Binary Relevance method
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on each label (activity) in a Binary Relevance setup
model.fit(X_train, y_train)

# Predict activities
y_pred = model.predict(X_test)

# Evaluate the model - calculate accuracy for each activity
accuracies = []
for i, label in enumerate(mlb.classes_):
    accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
    accuracies.append(accuracy)
    print(f"Accuracy for {label}: {accuracy:.2f}")

# Average accuracy
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy: {average_accuracy:.2f}")
