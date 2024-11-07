import pandas as pd
# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')
# Display the first few rows of the dataset
print(df.head())

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date
df = df.sort_values('Date')

# Extract features for anomaly detection
features = ['Revenue', 'Expense', 'Profit']  # Replace with relevant features
X = df[features]

from sklearn.ensemble import IsolationForest

# Initialize the Isolation Forest model with a different contamination parameter
model_if = IsolationForest(contamination=0.02, random_state=42)

# Fit the model to the data
model_if.fit(X)

# Predict anomalies
df['Anomaly_IF'] = model_if.predict(X)

# Anomalies are labeled as -1, normal data as 1
anomalies_if = df[df['Anomaly_IF'] == -1]
normal_data_if = df[df['Anomaly_IF'] == 1]

print(f'Number of anomalies detected by Isolation Forest: {len(anomalies_if)}')

from sklearn.neighbors import LocalOutlierFactor

# Initialize the Local Outlier Factor model
model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)

# Fit the model and predict anomalies
df['Anomaly_LOF'] = model_lof.fit_predict(X)

# Anomalies are labeled as -1, normal data as 1
anomalies_lof = df[df['Anomaly_LOF'] == -1]
normal_data_lof = df[df['Anomaly_LOF'] == 1]

print(f'Number of anomalies detected by LOF: {len(anomalies_lof)}')

from sklearn.svm import OneClassSVM

# Initialize the One-Class SVM model
model_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma=0.1)

# Fit the model to the data
model_svm.fit(X)

# Predict anomalies
df['Anomaly_SVM'] = model_svm.predict(X)

# Anomalies are labeled as -1, normal data as 1
anomalies_svm = df[df['Anomaly_SVM'] == -1]
normal_data_svm = df[df['Anomaly_SVM'] == 1]

print(f'Number of anomalies detected by One-Class SVM: {len(anomalies_svm)}')

from sklearn.svm import OneClassSVM

# Initialize the One-Class SVM model
model_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma=0.1)

# Fit the model to the data
model_svm.fit(X)

# Predict anomalies
df['Anomaly_SVM'] = model_svm.predict(X)

# Anomalies are labeled as -1, normal data as 1
anomalies_svm = df[df['Anomaly_SVM'] == -1]
normal_data_svm = df[df['Anomaly_SVM'] == 1]

print(f'Number of anomalies detected by One-Class SVM: {len(anomalies_svm)}')

import matplotlib.pyplot as plt

# Plot revenue with anomalies detected by Isolation Forest
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Revenue'], label='Revenue')
plt.scatter(anomalies_if['Date'], anomalies_if['Revenue'], color='red', label='Anomalies (IF)')
plt.title('Revenue with Anomalies Detected by Isolation Forest')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Plot revenue with anomalies detected by LOF
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Revenue'], label='Revenue')
plt.scatter(anomalies_lof['Date'], anomalies_lof['Revenue'], color='orange', label='Anomalies (LOF)')
plt.title('Revenue with Anomalies Detected by LOF')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Plot revenue with anomalies detected by One-Class SVM
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Revenue'], label='Revenue')
plt.scatter(anomalies_svm['Date'], anomalies_svm['Revenue'], color='green', label='Anomalies (SVM)')
plt.title('Revenue with Anomalies Detected by One-Class SVM')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

