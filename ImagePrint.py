import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Transcribing the data from the image to a pandas DataFrame
data = pd.read_csv('res_data_log.csv').iloc[:5000000, 1:]

# Creating the DataFrame
df = pd.DataFrame(data)

# Initialize lists to store metrics for plotting
r2_scores_lgbm_train, r2_scores_lgbm_test = [], []
mae_scores_lgbm_train, mae_scores_lgbm_test = [], []
mse_scores_lgbm_train, mse_scores_lgbm_test = [], []

r2_scores_mlp_train, r2_scores_mlp_test = [], []
mae_scores_mlp_train, mae_scores_mlp_test = [], []
mse_scores_mlp_train, mse_scores_mlp_test = [], []

# Iterate over different values of A from 1 to 10
for A in range(1, 11):
    print(f"\nA = {A}:")

    # Feature engineering: create lag features for the past A observations
    for i in range(A, A + 10):
        df[f'prey_lag_{i}'] = df['prey'].shift(i)
        df[f'predator_lag_{i}'] = df['predator'].shift(i)

    # Drop rows with NaN values created by shifting
    df = df.dropna()

    # Prepare the data for LightGBM
    target = 'predator'
    features = [f for f in df.columns if f not in ['time', target]]
    X = df[features]
    y = df[target]

    # Define the index to split the data into training and validation sets
    split_index = int(0.5 * len(df))  # 50% of the data for training

    # Split the data into training and validation sets
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    # Define LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbose': -1,
        'n_estimators': 1000
    }

    # Initialize the model
    model = lgb.LGBMRegressor(**params)

    # Train the model
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              eval_metric=['l2', 'l1'],
              #early_stopping_rounds=20,
              #verbose=False
              )

    # Predict on the training and validation sets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Calculate metrics for LightGBM
    r2_train_lgbm = r2_score(y_train, y_pred_train)
    r2_val_lgbm = r2_score(y_val, y_pred_val)
    mae_train_lgbm = mean_absolute_error(y_train, y_pred_train)
    mae_val_lgbm = mean_absolute_error(y_val, y_pred_val)
    mse_train_lgbm = mean_squared_error(y_train, y_pred_train)
    mse_val_lgbm = mean_squared_error(y_val, y_pred_val)

    # Store metrics for plotting
    r2_scores_lgbm_train.append(r2_train_lgbm)
    r2_scores_lgbm_test.append(r2_val_lgbm)
    mae_scores_lgbm_train.append(mae_train_lgbm)
    mae_scores_lgbm_test.append(mae_val_lgbm)
    mse_scores_lgbm_train.append(mse_train_lgbm)
    mse_scores_lgbm_test.append(mse_val_lgbm)

    # Initialize the MLPRegressor model
    model1 = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=1000, random_state=42)

    # Train the model
    model1.fit(X_train, y_train)  # Use X_train, y_train instead of X_train1, y_train1

    # Predict on the training and validation sets
    y_pred_train_mlp = model1.predict(X_train)
    y_pred_val_mlp = model1.predict(X_val)

    # Calculate metrics for MLPRegressor
    r2_train_mlp = r2_score(y_train, y_pred_train_mlp)
    r2_val_mlp = r2_score(y_val, y_pred_val_mlp)
    mae_train_mlp = mean_absolute_error(y_train, y_pred_train_mlp)
    mae_val_mlp = mean_absolute_error(y_val, y_pred_val_mlp)
    mse_train_mlp = mean_squared_error(y_train, y_pred_train_mlp)
    mse_val_mlp = mean_squared_error(y_val, y_pred_val_mlp)

    # Store metrics for plotting
    r2_scores_mlp_train.append(r2_train_mlp)
    r2_scores_mlp_test.append(r2_val_mlp)
    mae_scores_mlp_train.append(mae_train_mlp)
    mae_scores_mlp_test.append(mae_val_mlp)
    mse_scores_mlp_train.append(mse_train_mlp)
    mse_scores_mlp_test.append(mse_val_mlp)

    print("==================:", A)
    print("LightGBM - R² (Train):", r2_train_lgbm)
    print("LightGBM - R² (Validation):", r2_val_lgbm)
    print("LightGBM - MAE (Train):", mae_train_lgbm)
    print("LightGBM - MAE (Validation):", mae_val_lgbm)
    print("LightGBM - MSE (Train):", mse_train_lgbm)
    print("LightGBM - MSE (Validation):", mse_val_lgbm)


    # Plot and save figures
    plt.figure(figsize=(10,20))

    # R2 Score Plot
    plt.subplot(3, 1, 1)
    plt.plot(range(1, A + 1), r2_scores_lgbm_train, label='LightGBM Train R²')
    plt.plot(range(1, A + 1), r2_scores_lgbm_test, label='LightGBM Validation R²')
    plt.xlabel('A')
    plt.ylabel('R²')
    plt.title('R² Score')
    plt.legend()
    #plt.savefig(f'R2_Score_A_{A}.png')

    # MAE Score Plot
    plt.subplot(3, 1, 2)
    plt.plot(range(1, A + 1), mae_scores_lgbm_train, label='LightGBM Train MAE')
    plt.plot(range(1, A + 1), mae_scores_lgbm_test, label='LightGBM Validation MAE')
    plt.xlabel('A')
    plt.ylabel('MAE')
    plt.title('MAE Score')
    plt.legend()
    #plt.savefig(f'MAE_Score_A_{A}.png')

    # MSE Score Plot
    plt.subplot(3, 1, 3)
    plt.plot(range(1, A + 1), mse_scores_lgbm_train, label='LightGBM Train MSE')
    plt.plot(range(1, A + 1), mse_scores_lgbm_test, label='LightGBM Validation MSE')
    plt.xlabel('A')
    plt.ylabel('MSE')
    plt.title('MSE Score')
    plt.legend()

    plt.savefig(f'LGM_Score_A_{A}.png')

    # Clear the plot
    plt.clf()

    #MLP
    print("MLPRegressor - R² (Train):", r2_train_mlp)
    print("MLPRegressor - R² (Validation):", r2_val_mlp)
    print("MLPRegressor - MAE (Train):", mae_train_mlp)
    print("MLPRegressor - MAE (Validation):", mae_val_mlp)
    print("MLPRegressor - MSE (Train):", mse_train_mlp)
    print("MLPRegressor - MSE (Validation):", mse_val_mlp)

    #MLP_r2
    plt.figure(figsize=(10,20))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, A + 1), r2_scores_mlp_train, label='MLPRegressor Train R²')
    plt.plot(range(1, A + 1), r2_scores_mlp_test, label='MLPRegressor Validation R²')
    plt.xlabel('A')
    plt.ylabel('R²')
    plt.title('R² Score')
    plt.legend()

    #MAE
    plt.subplot(3, 1, 2)
    plt.plot(range(1, A + 1), mae_scores_mlp_train, label='MLPRegressor Train MAE')
    plt.plot(range(1, A + 1), mae_scores_mlp_test, label='MLPRegressor Validation MAE')
    plt.xlabel('A')
    plt.ylabel('MAE')
    plt.title('MAE Score')
    plt.legend()

    #MSE
    plt.subplot(3, 1, 3)
    plt.plot(range(1, A + 1), mse_scores_mlp_train, label='MLPRegressor Train MSE')
    plt.plot(range(1, A + 1), mse_scores_mlp_test, label='MLPRegressor Validation MSE')
    plt.xlabel('A')
    plt.ylabel('MSE')
    plt.title('MSE Score')
    plt.legend()

    plt.savefig(f'MLP_Score_A_{A}.png')

    plt.clf()





