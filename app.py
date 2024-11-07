from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
features = ['Revenue', 'Expense', 'Profit']
X = df[features]

# Initialize anomaly detection models
model_if = IsolationForest(contamination=0.02, random_state=42)
model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
model_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma=0.1)

# Fit the models
model_if.fit(X)
df['Anomaly_IF'] = model_if.predict(X)
df['Anomaly_LOF'] = model_lof.fit_predict(X)
model_svm.fit(X)
df['Anomaly_SVM'] = model_svm.predict(X)

# Segregate anomaly data
anomaly_dir = 'anomaly_data'
if not os.path.exists(anomaly_dir):
    os.makedirs(anomaly_dir)
anomalies_if = df[df['Anomaly_IF'] == -1]
anomalies_lof = df[df['Anomaly_LOF'] == -1]
anomalies_svm = df[df['Anomaly_SVM'] == -1]
anomalies_if.to_csv(os.path.join(anomaly_dir, 'anomalies_if.csv'), index=False)
anomalies_lof.to_csv(os.path.join(anomaly_dir, 'anomalies_lof.csv'), index=False)
anomalies_svm.to_csv(os.path.join(anomaly_dir, 'anomalies_svm.csv'), index=False)

# Initialize other models
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)
df['Predicted_Revenue'] = model_xgb.predict(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

df['High_Revenue'] = (df['Revenue'] > df['Revenue'].median()).astype(int)
y_class = df['High_Revenue']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_class, y_train_class)
df['Predicted_High_Revenue'] = model_rf.predict(X)

model_dl = Sequential()
model_dl.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_dl.add(Dense(32, activation='relu'))
model_dl.add(Dense(1, activation='linear'))
model_dl.compile(optimizer='adam', loss='mean_squared_error')
model_dl.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
df['Predicted_Revenue_DL'] = model_dl.predict(X)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    response = generate_response(user_input)
    return jsonify({'response': response})

def generate_response(user_input):
    if 'anomaly' in user_input.lower():
        return "We use advanced anomaly detection techniques like Isolation Forest, Local Outlier Factor, and One-Class SVM to identify and segregate false or corrupted data. The anomalies are saved in separate files for further analysis."
    elif 'financial data' in user_input.lower():
        return "Our financial data management system leverages AI to enhance data quality, perform predictive analytics, and improve decision-making processes."
    elif 'predict revenue' in user_input.lower():
        return f"The predicted revenue using XGBoost is {df['Predicted_Revenue'].iloc[-1]}, and using Deep Learning is {df['Predicted_Revenue_DL'].iloc[-1]}."
    elif 'cluster' in user_input.lower():
        return f"The data has been clustered into {df['Cluster'].nunique()} clusters using K-Means."
    else:
        return "I'm here to help with financial data management and AI enhancements. Ask me anything related to these topics!"

if __name__ == '__main__':
    app.run(debug=True)

