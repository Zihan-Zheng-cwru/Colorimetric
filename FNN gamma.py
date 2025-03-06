import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
train_data = pd.read_excel('/Users/zihanzheng/Desktop/Code/gamma/gamma_train.xlsx')
test_data = pd.read_excel('/Users/zihanzheng/Desktop/Code/gamma/gamma_test.xlsx')

# Extracting features and responses
X_train = train_data.iloc[:, 1:23]
y_train = train_data.iloc[:, 23:]
X_test = test_data.iloc[:, 1:23]
y_test = test_data.iloc[:, 23:]

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Model
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))# Input layer + Hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))# Another hidden layer
model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Fit the model
model.fit(X_train_scaled, y_train, epochs=400, batch_size=8, verbose=1)

# Predictions on the test set
y_pred_nn = model.predict(X_test_scaled)

# Calculate RMSE for both response columns
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn, multioutput='raw_values'))
print("RMSE on Test Set for each response variable with Neural Network:", rmse_nn)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(y_pred_nn, columns=['Predicted_Response1', 'Predicted_Response2'])

# Optionally, add any additional columns you need, such as identifiers from the test set
predictions_df['Index'] = test_data['Index']

# Output the predictions to an Excel file
output_file_path = '/Users/zihanzheng/Desktop/Code/gamma/FNN_predictions.xlsx'
predictions_df.to_excel(output_file_path, index=False)


print(f"Predictions saved to {output_file_path}")