import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('D:\\Main project\\household_power_consumption.csv\\household_power_consumption.csv')

# Convert date and time to datetime, handle different formats
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True, errors='coerce')

# Drop original Date and Time columns
data.drop(columns=['Date', 'Time'], inplace=True)

# Replace non-numeric values with NaN
data.replace('?', np.nan, inplace=True)

# Drop rows with NaN values
data.dropna(inplace=True)

# Convert all columns to numeric (coerce errors to NaN and then drop NaN values)
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# Define features and target
X = data.drop(columns=['Global_active_power'])  # Features
y = data['Global_active_power']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=200, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')

# Make predictions
y_pred_ann = model.predict(X_test_scaled)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred_ann)
print(f'MAPE: {mape}')

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_ann, label='Predicted')
plt.legend()
plt.show()
