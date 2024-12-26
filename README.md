# anomaly-detection-in-network-traffic-data


1. Simulate Network Traffic Data:
The code generates synthetic data for normal and anomalous network traffic.
np.random.normal is used to create 10,000 samples of normal traffic with a mean of 0 and a standard deviation of 1, and 500 anomalous samples with a mean of 5 and a standard deviation of 2.
The normal and anomalous data are combined into one dataset, and corresponding labels are created (1 for normal and -1 for anomalous).
2. Preprocess the Data:
The StandardScaler is used to standardize the data (mean = 0, variance = 1) for better model performance.
PCA (Principal Component Analysis) is applied to reduce the dataset to 2 dimensions, which is useful for both visualization and improving model performance.
3. Train the Anomaly Detection Model:
The code uses the IsolationForest model from sklearn, which is an unsupervised learning algorithm used to detect anomalies.
The model is trained on the scaled data, where n_estimators=100 sets the number of trees in the forest and contamination=0.05 indicates the proportion of outliers in the data.
4. Predict Anomalies:
The model predicts whether each data point is normal (1) or an anomaly (-1) based on the trained model.
5. Evaluate the Model Performance:
The model's performance is evaluated using classification_report, which shows metrics such as precision, recall, and F1-score, comparing the predicted anomalies to the actual labels.
6. Visualize the Results:
The code visualizes the 2D reduced data (after PCA) using matplotlib. The points are colored based on whether they are predicted to be normal or anomalous, allowing for a visual inspection of the anomalies.
7. Save the Trained Model:
The trained IsolationForest model, the scaler, and the PCA transformation are saved using joblib so they can be loaded and used later without retraining.
8. Simulate Real-Time Data Processing:
A function process_streaming_data simulates real-time anomaly detection by processing new batches of data. The new data is standardized and reduced using PCA, and anomalies are detected by the pre-trained model.
A new batch of synthetic network traffic is generated, and anomalies in the batch are predicted and printed.
