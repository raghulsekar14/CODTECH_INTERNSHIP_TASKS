import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import graphviz
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset: Kaggle Titanic Dataset (Classification)
csv_path = os.path.join(BASE_DIR, "titanic.csv")
# 1. Load the dataset
data = pd.read_csv(csv_path)

# Select useful columns
# Target: Survived (0 = No, 1 = Yes)
features = ['Pclass', 'Sex', 'Age', 'Fare']
target = 'Survived'

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())

# Encode categorical data
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

X = data[features]
y = data[target]

feature_names = features
class_names = ['Not Survived', 'Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualize the Decision Tree
plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True
)

graph = graphviz.Source(dot_data)
graph

rules = export_text(model, feature_names=feature_names)
print(rules)

# Sample prediction
sample = pd.DataFrame([[3, 0, 18, 7.2292]], columns=features)
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")
