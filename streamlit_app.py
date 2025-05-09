import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load Titanic dataset
try:
    data = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the dataset exists in the same directory.")
    exit()

# Preprocess the data
try:
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
except KeyError as e:
    print(f"Error: Missing expected column in the dataset - {e}")
    exit()

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

if not all(col in data.columns for col in features + [target]):
    print("Error: One or more required columns are missing in the dataset.")
    exit()

X = data[features]
y = data[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_file = 'titanic_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

if os.path.exists(model_file):
    print(f"Model successfully saved to '{model_file}'")
else:
    print("Error: Failed to save the model.")
