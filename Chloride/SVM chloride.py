import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_excel(file_path, sheet_name='model')

# Selecting the input features and the target variable
X = data[['R', 'G', 'B', 'H', 'S', 'I', 'Lux']]
y = data['Con']

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Store results
r2_train_scores = []
r2_test_scores = []
r2_val_scores = []
rmse_train_scores = []
rmse_test_scores = []
rmse_val_scores = []
predictions = []

for i in range(10):
    # Randomly split the data into training (60%), testing (20%), and validation (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=None)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=None)

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_val_pred = best_model.predict(X_val)

    # Store the actual and predicted values for validation set
    predictions.append(pd.DataFrame({
        'Run': i+1,
        'Actual': y_val,
        'Predicted': y_val_pred
    }))

    # Calculate R2 and RMSE for training, testing, and validation sets
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Store the scores
    r2_train_scores.append(r2_train)
    rmse_train_scores.append(rmse_train)
    r2_test_scores.append(r2_test)
    rmse_test_scores.append(rmse_test)
    r2_val_scores.append(r2_val)
    rmse_val_scores.append(rmse_val)

    # Print the R2 and RMSE for this run
    print(f"Run {i+1}:")
    print(f"  Training - R2: {r2_train:.4f}, RMSE: {rmse_train:.4f}")
    print(f"  Testing  - R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}")
    print(f"  Validation - R2: {r2_val:.4f}, RMSE: {rmse_val:.4f}")

# Calculate the mean and standard deviation of R2 and RMSE for training, testing, and validation sets
r2_train_mean = np.mean(r2_train_scores)
r2_train_std = np.std(r2_train_scores)
rmse_train_mean = np.mean(rmse_train_scores)
rmse_train_std = np.std(rmse_train_scores)

r2_test_mean = np.mean(r2_test_scores)
r2_test_std = np.std(r2_test_scores)
rmse_test_mean = np.mean(rmse_test_scores)
rmse_test_std = np.std(rmse_test_scores)

r2_val_mean = np.mean(r2_val_scores)
r2_val_std = np.std(r2_val_scores)
rmse_val_mean = np.mean(rmse_val_scores)
rmse_val_std = np.std(rmse_val_scores)

# Print the mean and standard deviation of R2 and RMSE
print("\nSummary:")
print(f"Training - Mean R2: {r2_train_mean:.4f}, Std R2: {r2_train_std:.4f}, Mean RMSE: {rmse_train_mean:.4f}, Std RMSE: {rmse_train_std:.4f}")
print(f"Testing  - Mean R2: {r2_test_mean:.4f}, Std R2: {r2_test_std:.4f}, Mean RMSE: {rmse_test_mean:.4f}, Std RMSE: {rmse_test_std:.4f}")
print(f"Validation - Mean R2: {r2_val_mean:.4f}, Std R2: {r2_val_std:.4f}, Mean RMSE: {rmse_val_mean:.4f}, Std RMSE: {rmse_val_std:.4f}")

# Combine all predictions into a single DataFrame
predictions_df = pd.concat(predictions, ignore_index=True)

# Save all the results in an Excel file
output_file_path = '/Users/zihanzheng/Desktop/Project 3/Chloride_Results_Predictions.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
    summary_df = pd.DataFrame({
        'Metric': ['R2 Mean (Train)', 'R2 Std (Train)', 'RMSE Mean (Train)', 'RMSE Std (Train)',
                   'R2 Mean (Test)', 'R2 Std (Test)', 'RMSE Mean (Test)', 'RMSE Std (Test)',
                   'R2 Mean (Val)', 'R2 Std (Val)', 'RMSE Mean (Val)', 'RMSE Std (Val)'],
        'Value': [r2_train_mean, r2_train_std, rmse_train_mean, rmse_train_std,
                  r2_test_mean, r2_test_std, rmse_test_mean, rmse_test_std,
                  r2_val_mean, r2_val_std, rmse_val_mean, rmse_val_std]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

# Plot the actual vs. predicted values for training, testing, and validation sets
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set: Actual vs Predicted')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Set: Actual vs Predicted')

plt.subplot(1, 3, 3)
plt.scatter(y_val, y_val_pred, alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Validation Set: Actual vs Predicted')

plt.tight_layout()
plt.show()
