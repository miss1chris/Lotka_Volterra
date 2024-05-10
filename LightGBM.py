import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Transcribing the data from the image to a pandas DataFrame
data = pd.read_csv('res_data_log.csv').iloc[:,1:]

# Creating the DataFrame
df = pd.DataFrame(data)

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
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=20,
              verbose=False)

    # Predict on the validation set
    y_pred = model.predict(X_val)

    # Initialize the MLPRegressor model
    model1 = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=1000, random_state=42)

    # Train the model
    model1.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)
    # Calculate R² and MAE
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("==================:", A)
    print("R²:", r2)
    print("MAE:", mae)
    # Predict on the validation set
    y_pred1 = model1.predict(X_val)
    # Calculate R² and MAE

    r2 = r2_score(y_val, y_pred1)
    mae = mean_absolute_error(y_val, y_pred1)
    print("==================:", A)
    print("R²:", r2)
    print("MAE:", mae)