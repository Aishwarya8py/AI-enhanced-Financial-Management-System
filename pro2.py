import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

#Data Processing
# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date
df = df.sort_values('Date')

# Extract features for machine learning
features = ['Revenue', 'Expense', 'Profit']  # Replace with relevant features
X = df[features]

#ADVANCED MACHINE LEARNING MODELS
#XGBoost for Regression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
y = df['Revenue']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)

# Make predictions
predictions_xgb = model_xgb.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions_xgb)
print(f'Mean Squared Error (XGBoost): {mse}')

#k-MEANS FOR CLUSTERING
from sklearn.cluster import KMeans

# Initialize the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

print(df.head())


#RANDOM FOREST FOR CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a binary target variable for classification (e.g., high vs. low revenue)
df['High_Revenue'] = (df['Revenue'] > df['Revenue'].median()).astype(int)
y_class = df['High_Revenue']

# Split the data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_class, y_train_class)

# Make predictions
predictions_rf = model_rf.predict(X_test_class)

# Evaluate the model
accuracy = accuracy_score(y_test_class, predictions_rf)
print(f'Accuracy (Random Forest): {accuracy}')

#Deep Learning Model fpr Complex Prediction

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Prepare the data
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the deep learning model
model_dl = Sequential()
model_dl.add(Dense(64, input_dim=X_train_dl.shape[1], activation='relu'))
model_dl.add(Dense(32, activation='relu'))
model_dl.add(Dense(1, activation='linear'))

# Compile the model
model_dl.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_dl.fit(X_train_dl, y_train_dl, epochs=50, batch_size=10, verbose=1)

# Make predictions
predictions_dl = model_dl.predict(X_test_dl)

# Evaluate the model
mse_dl = mean_squared_error(y_test_dl, predictions_dl)
print(f'Mean Squared Error (Deep Learning): {mse_dl}')


#Cross Verification
# Cross-verify XGBoost predictions with Random Forest classification
df['Predicted_Revenue'] = model_xgb.predict(X)
df['Predicted_High_Revenue'] = model_rf.predict(X)

# Compare predicted high revenue with actual high revenue
cross_verification = (df['High_Revenue'] == df['Predicted_High_Revenue']).mean()
print(f'Cross-Verification Accuracy: {cross_verification}')

#Visualize the Results
import matplotlib.pyplot as plt

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Date'], df['Revenue'], c=df['Cluster'], cmap='viridis')
plt.title('Revenue Clusters')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()

# Plot actual vs. predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Revenue'], label='Actual Revenue')
plt.plot(df['Date'], df['Predicted_Revenue'], label='Predicted Revenue', linestyle='--')
plt.title('Actual vs. Predicted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

