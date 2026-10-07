import numpy as np
import pandas as pd

# Read datasets
df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

# Combine datasets
dataset = pd.concat([df1, df2, df3], ignore_index=True)

# Remove spaces from column names
dataset.columns = dataset.columns.str.strip()

# Separate features and target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Encode target labels
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# Classification algorithm
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(
    accuracy_score(y_test, y_pred)
))

print("\nPrecision : {:.4f}".format(
    precision_score(y_test, y_pred, average='weighted', zero_division=0)
))

print("\nRecall : {:.4f}".format(
    recall_score(y_test, y_pred, average='weighted', zero_division=0)
))

print("\nF1 Score : {:.4f}".format(
    f1_score(y_test, y_pred, average='weighted', zero_division=0)
))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))