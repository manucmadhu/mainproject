import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
def create_model(optimizer='adam', dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(512, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(4, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

optimizer = Adam(learning_rate=0.001)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * np.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Create and train the model
ann_model = create_model(optimizer=optimizer)
history = ann_model.fit(X_train_scaled, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

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
# accuracy of 81.65%

# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# Load the pre-trained model
model = load_model('power_management_ann_model.h5')

# Data preprocessing components
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Pre-compute mean and std values for scaling
# Here, you should compute these values from your training data
# or save them along with the model and load them if you have them.
# For demonstration purposes, assuming values are set manually.
# Replace these with actual values from your training preprocessing.

mean_values = np.array([20.0, 50.0, 10.0, 30.0, 15.0])  # Replace with actual mean values
std_values = np.array([5.0, 10.0, 2.0, 8.0, 4.0])       # Replace with actual std values

def preprocess_input(user_input):
    # Convert user input to DataFrame
    df_input = pd.DataFrame([user_input], columns=['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows'])
    
    # Handle missing values
    df_input_imputed = imputer.fit_transform(df_input)
    
    # Scale features
    df_input_scaled = (df_input_imputed - mean_values) / std_values
    
    return df_input_scaled

def predict_power_consumption(user_input):
    # Preprocess the user input
    processed_input = preprocess_input(user_input)
    
    # Make predictions
    predictions = model.predict(processed_input)
    
    # Return the predictions
    return predictions[0]

def get_user_input():
    # Collect user input
    temperature = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    wind_speed = float(input("Enter Wind Speed: "))
    general_diffuse_flows = float(input("Enter General Diffuse Flows: "))
    diffuse_flows = float(input("Enter Diffuse Flows: "))
    
    # Return as a list
    return [temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows]

# Example usage
if __name__ == "__main__":
    # Get user input
    user_input = get_user_input()
    
    # Get predictions
    predictions = predict_power_consumption(user_input)
    
    print("Predicted Power Consumption:")
    print(f"Zone 1: {predictions[0]:.2f}")
    print(f"Zone 2: {predictions[1]:.2f}")
    print(f"Zone 3: {predictions[2]:.2f}")
    print(f"Total: {predictions[3]:.2f}")

