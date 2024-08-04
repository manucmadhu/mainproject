import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('D:/Main project/powerconsumption.csv')

data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')

# Feature and target columns
features = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]
target = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3', 'Total']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Data Preprocessing
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Define the model
ann_model = Sequential()
ann_model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(128, activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(64, activation='relu'))
ann_model.add(Dense(4, activation='linear'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
ann_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = ann_model.fit(X_train_scaled, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, mae = ann_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")

# Make predictions with the ANN model
predictions = ann_model.predict(X_test_scaled)

# Compare predictions with actual values
plt.figure(figsize=(14, 7))
for i, target_col in enumerate(target.columns):
    plt.plot(y_test[target_col].values, label=f'Actual {target_col}')
    plt.plot(predictions[:, i], label=f'Predicted {target_col}')
plt.legend()
plt.show()

# Save the ANN model
ann_model.save('power_management_ann_model.h5')

# Load the ANN model
loaded_ann_model = load_model('power_management_ann_model.h5')

# Calculate and print accuracy for ANN model
from sklearn.metrics import mean_absolute_percentage_error
y_pred_ann = loaded_ann_model.predict(X_test_scaled)
mape = mean_absolute_percentage_error(y_test, y_pred_ann)
accuracy = 100 - mape * 100
print(f"Neural Network Accuracy: {accuracy:.2f}%")
