import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump, load
import time

# Step 1: Simulate Network Traffic Data (This should be replaced by real network data)
np.random.seed(42)
normal_traffic = np.random.normal(0, 1, (10000, 5))  # Simulating larger 10,000 samples, 5 features
anomalous_traffic = np.random.normal(5, 2, (500, 5))  # 500 anomalous samples

# Combine normal and anomalous traffic data
data = np.vstack([normal_traffic, anomalous_traffic])
labels = np.concatenate([np.ones(len(normal_traffic)), -1*np.ones(len(anomalous_traffic))])  # 1 = Normal, -1 = Anomaly

# Step 2: Preprocess the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Optional: Reduce dimensions for visualization and better performance with PCA
pca = PCA(n_components=2)  # Reduce to 2D for visualization
data_reduced = pca.fit_transform(data_scaled)

# Step 3: Train the Anomaly Detection Model (Using Isolation Forest)
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)  # n_jobs=-1 uses all cores
model.fit(data_scaled)

# Step 4: Predict Anomalies (1 for normal, -1 for anomaly)
predictions = model.predict(data_scaled)

# Step 5: Evaluate Model Performance (Optional: If you have labeled data)
print("\nClassification Report:")
print(classification_report(labels, predictions))

# Step 6: Visualize the Results (For 2D representation)
plt.figure(figsize=(10, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=predictions, cmap='coolwarm', alpha=0.6)
plt.title("Anomaly Detection in Network Traffic (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Prediction")
plt.show()

# Step 7: Save the Trained Model for Future Use
dump(model, 'anomaly_detection_model.joblib')
dump(scaler, 'scaler.joblib')
dump(pca, 'pca_model.joblib')

# Step 8: Real-Time Simulation (If you're using streaming data)
# Here, we simulate real-time processing using batches
def process_streaming_data(new_data):
    # Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    new_data_reduced = pca.transform(new_data_scaled)
    
    # Predict anomalies in the new batch
    new_predictions = model.predict(new_data_scaled)
    
    # Report anomalies
    anomalies = new_data[new_predictions == -1]
    print(f"Detected {len(anomalies)} anomalies in the new data batch.")
    
    # Optionally, log or alert security teams if needed
    return anomalies

# Simulate a new batch of network data coming in
time.sleep(2)  # Simulating delay in receiving new data
new_batch = np.random.normal(0, 1, (500, 5))  # Simulating new network traffic batch
anomalies = process_streaming_data(new_batch)

