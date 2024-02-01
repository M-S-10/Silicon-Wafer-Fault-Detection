import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('wafer_train_01.csv')  # Replace 'your_training_dataset.csv' with the actual file name

# Drop the first row (header with label names)
df = df.drop(0)

# Extract features (sensor values) and target variable (good/bad)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].astype(int).values

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the KNN classifier with adjusted parameters
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)  # Experiment with p values
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Initialize a Decision Tree classifier as an alternative
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the performance of KNN classifier
labels = np.unique(y_test)
print("KNN Classifier:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn, labels=labels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, zero_division=1, labels=labels))

# Evaluate the performance of Decision Tree classifier
print("\nDecision Tree Classifier:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt, labels=labels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, zero_division=1, labels=labels))

# Generate confusion matrix heatmap for KNN classifier
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_knn, labels=labels), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - KNN Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Generate confusion matrix heatmap for Decision Tree classifier
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_dt, labels=labels), annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
