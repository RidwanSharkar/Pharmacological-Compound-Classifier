# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data including activities
data = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\compound_info.csv')

# Handling missing values and preparing target
data.dropna(subset=['Activities'], inplace=True)  # Drop entries without activities
mlb = MultiLabelBinarizer()
activities_encoded = mlb.fit_transform(data['Activities'].apply(eval))

# Prepare features
features = data[['NumHBD', 'NumHBA', 'ExactMW', 'AMW']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, activities_encoded, test_size=0.2, random_state=42)

# Initialize Binary Relevance Model with Random Forest
classifiers = []
for i in range(activities_encoded.shape[1]):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train[:, i])
    classifiers.append(clf)

# Prediction and Evaluation
for idx, clf in enumerate(classifiers):
    y_pred = clf.predict(X_test)
    print(f"Accuracy for {mlb.classes_[idx]}: {accuracy_score(y_test[:, idx], y_pred)}")


"""
# 1) DATA CLEANING:
data = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Psychoactive-Compounds-Analysis\\data\\psychoactive compounds.csv')
data.isnull().sum()
data = data.dropna()  # Example: Dropping rows with missing values

# 2) FEATURE SELECTION:
features = data[['NumHBD', 'NumHBA', 'ExactMW', 'AMW']]
target = data['Activity']  # after scrape MoA

# 3) MODEL:
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# For Regression (predicting a continuous value like potency):
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# For Classification (predicting a category like toxic/non-toxic):
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 4) EVALUATION:
y_pred = model.predict(X_test)
# For Regression
print(mean_squared_error(y_test, y_pred))
# For Classification
print(accuracy_score(y_test, y_pred))
"""
