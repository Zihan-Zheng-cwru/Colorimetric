import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the data
data = pd.read_excel(file_path, sheet_name='model')

# Selecting the input features and the target variable
X = data[['R', 'G', 'B', 'H', 'S', 'I', 'Grayscale', 'Lux']]
y = data['Con']
groups = data['response_group']  # Column that identifies the response groups

# Ensure groups have at least 2 members
group_counts = groups.value_counts()
valid_groups = group_counts[group_counts >= 2].index
valid_indices = groups.isin(valid_groups)

X = X[valid_indices]
y = y[valid_indices]
groups = groups[valid_indices]

# Convert y to a numpy array
y = y.values

# Define the parameter space for Random Search
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.1, 0.2, 0.4],
    'colsample_bytree': [0.4, 0.6, 0.800]
}

# Metrics storage
train_r2_scores, train_rmse_scores = [], []
val_r2_scores, val_rmse_scores = [], []
test_r2_scores, test_rmse_scores = [], []
best_params_list = []

# Store the last run predictions for plotting and saving
last_train_predictions, last_val_predictions, last_test_predictions = None, None, None
last_y_train, last_y_val, last_y_test = None, None, None

# DataFrames to store combined data
train_combined_df, val_combined_df, test_combined_df = None, None, None

# Run 10 times with different random states
for i in range(10):
    print(f"Iteration {i + 1}")

    # Use GroupShuffleSplit to create train/val/test splits while preserving group structure
    gss_train_val = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=i)
    train_idx, val_test_idx = next(gss_train_val.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val_test, y_val_test = X.iloc[val_test_idx], y[val_test_idx]
    groups_val_test = groups.iloc[val_test_idx]

    # Split val_test into validation and test sets
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=i)
    val_idx, test_idx = next(gss_val_test.split(X_val_test, y_val_test, groups=groups_val_test))

    X_val, X_test = X_val_test.iloc[val_idx], X_val_test.iloc[test_idx]
    y_val, y_test = y_val_test[val_idx], y_val_test[test_idx]

    # Create the XGBoost model
    xgb_model = XGBRegressor(random_state=42)

    # Perform Random Search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=30,  # number of parameter settings sampled
        cv=5,  # 5-fold CV
        verbose=0,
        random_state=42,
        n_jobs=2
    )
    random_search.fit(X_train, y_train)

    # Best model after Random Search
    best_xgb = random_search.best_estimator_

    # Store the best hyperparameters
    best_params = random_search.best_params_
    best_params_list.append(best_params)

    # Make predictions on all sets
    train_predictions = best_xgb.predict(X_train)
    val_predictions = best_xgb.predict(X_val)
    test_predictions = best_xgb.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    val_r2 = r2_score(y_val, val_predictions)
    val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

    # Store the scores
    train_r2_scores.append(train_r2);
    train_rmse_scores.append(train_rmse)
    val_r2_scores.append(val_r2);
    val_rmse_scores.append(val_rmse)
    test_r2_scores.append(test_r2);
    test_rmse_scores.append(test_rmse)

    # Combine the data for the last run
    if i == 9:
        last_train_predictions, last_y_train = train_predictions, y_train
        last_val_predictions, last_y_val = val_predictions, y_val
        last_test_predictions, last_y_test = test_predictions, y_test

        train_combined_df = pd.concat([X.iloc[train_idx].reset_index(drop=True),
                                       pd.DataFrame({'Actual': y_train, 'Predicted': train_predictions})], axis=1)

        val_combined_df = pd.concat([X_val.reset_index(drop=True),
                                     pd.DataFrame({'Actual': y_val, 'Predicted': val_predictions})], axis=1)

        test_combined_df = pd.concat([X_test.reset_index(drop=True),
                                      pd.DataFrame({'Actual': y_test, 'Predicted': test_predictions})], axis=1)

# Calculate mean and standard deviation of the evaluation metrics
mean_train_r2, std_train_r2 = np.mean(train_r2_scores), np.std(train_r2_scores)
mean_train_rmse, std_train_rmse = np.mean(train_rmse_scores), np.std(train_rmse_scores)

mean_val_r2, std_val_r2 = np.mean(val_r2_scores), np.std(val_r2_scores)
mean_val_rmse, std_val_rmse = np.mean(val_rmse_scores), np.std(val_rmse_scores)

mean_test_r2, std_test_r2 = np.mean(test_r2_scores), np.std(test_r2_scores)
mean_test_rmse, std_test_rmse = np.mean(test_rmse_scores), np.std(test_rmse_scores)

# Plotting scatter plots of actual vs. predicted values for the last run
plt.figure(figsize=(6, 18))

plt.subplot(3, 1, 1)
sns.scatterplot(x=last_y_train, y=last_train_predictions)
plt.title('Training Data: Actual vs. Predicted')
plt.xlabel('Actual Values');
plt.ylabel('Predicted Values')
plt.plot([last_y_train.min(), last_y_train.max()], [last_y_train.min(), last_y_train.max()], 'r--',
         label='Perfect Prediction')
plt.legend()

plt.subplot(3, 1, 2)
sns.scatterplot(x=last_y_val, y=last_val_predictions)
plt.title('Validation Data: Actual vs. Predicted')
plt.xlabel('Actual Values');
plt.ylabel('Predicted Values')
plt.plot([last_y_val.min(), last_y_val.max()], [last_y_val.min(), last_y_val.max()], 'r--', label='Perfect Prediction')
plt.legend()

plt.subplot(3, 1, 3)
sns.scatterplot(x=last_y_test, y=last_test_predictions)
plt.title('Testing Data: Actual vs. Predicted')
plt.xlabel('Actual Values');
plt.ylabel('Predicted Values')
plt.plot([last_y_test.min(), last_y_test.max()], [last_y_test.min(), last_y_test.max()], 'r--',
         label='Perfect Prediction')
plt.legend()

plt.tight_layout()
plt.show()

# Output results
print("Train - R2: {:.3f} ± {:.3f}, RMSE: {:.3f} ± {:.3f}".format(mean_train_r2, std_train_r2, mean_train_rmse,
                                                                  std_train_rmse))
print("Val   - R2: {:.3f} ± {:.3f}, RMSE: {:.3f} ± {:.3f}".format(mean_val_r2, std_val_r2, mean_val_rmse, std_val_rmse))
print("Test  - R2: {:.3f} ± {:.3f}, RMSE: {:.3f} ± {:.3f}".format(mean_test_r2, std_test_r2, mean_test_rmse,
                                                                  std_test_rmse))

# Print best hyperparameters for each run
for I, params in enumerate(best_params_list):
    print(f"Run {I + 1} - Best Hyperparameters: {params}")

# Save the combined data to an Excel file
output_path = '/Users/zihanzheng/Desktop/Combined_Predictions.xlsx'
with pd.ExcelWriter(output_path) as writer:
    train_combined_df.to_excel(writer, sheet_name='Train', index=False)
    val_combined_df.to_excel(writer, sheet_name='Validation', index=False)
    test_combined_df.to_excel(writer, sheet_name='Test', index=False)
