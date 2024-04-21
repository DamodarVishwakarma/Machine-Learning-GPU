import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text

# import tensorflow as tf

# import torch
# import torch.nn as nn
# import torch.optim as optim

from version_1.constants_v4 import *

import logging

log_filename = 'logs/modeling_endpoint_api.log'

# Clear the log file
# with open(log_filename, 'w') as file:
#     pass

# Setup logging
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger()

def convert_datetime_to_details(X_df):
    # Convert Datetime into individual features
    X_df['year'] = X_df['Datetime'].dt.year
    X_df['month'] = X_df['Datetime'].dt.month
    X_df['day'] = X_df['Datetime'].dt.day
    X_df['dayofweek'] = X_df['Datetime'].dt.dayofweek
    X_df['is_weekend'] = X_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Drop the original Datetime column
    X_df.drop('Datetime', axis=1, inplace=True)
    
    return X_df

def convert_features_columns(df, top_k):
    # Split out the features column into separate columns
    features_df = pd.json_normalize(df['features'])
    features_df.columns = [col for col in features_df.columns]

    return features_df[top_k]

def train_user_model(data):
    user_id, strategy_id, feature_set_name, model_name, hyperparams, phase, df, top_k = data
    
    # Check model's preferred device
    preferred_device = model_device_mapping.get(model_name, "CPU")
    
    library = model_name.split('_')[0]  # Assuming the model_name is in the format 'Library_ModelType'
     
    if library == "PyTorch":
        # import torch
        device = get_device_info(model_name)
        metrics, y_pred, used_y, used_data = start_training(model_name, hyperparams, df, top_k, phase, device=device)

    elif library == "Sklearn":
        # Scikit-learn does not have explicit GPU support but can utilize multi-core CPUs
        metrics, y_pred, used_y, used_data = start_training(model_name, hyperparams, df, top_k, phase)
    
    else:
        raise ValueError(f"Unsupported library: {library}")
    
    return metrics, y_pred, used_y, used_data

# Update the start_training function
def start_training(model_name, hyperparams, df, top_k, phase="train", device='cpu'):
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, X_val_full, X_test_full = prepare_data(df, top_k)

    if phase == "train":
        used_data = X_train
        used_data_full = X_train_full
        used_y = y_train
    elif phase == "validate":
        used_data = X_val
        used_data_full = X_val_full
        used_y = y_val
    elif phase == "test":
        used_data = X_test
        used_data_full = X_test_full
        used_y = y_test
    else:
        raise ValueError(f"Invalid phase: {phase}")

    library, ModelClass = MODEL_MAPPER.get(model_name)
    if not ModelClass:
        raise ValueError(f"Unsupported model: {model_name}")

    model_instance = ModelClass(**hyperparams)

    # If the model is PyTorch-based:
    if library == "PyTorch":
        model_instance.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_instance.parameters())
        
        # Convert data to PyTorch tensors
        X_train_torch = torch.from_numpy(X_train.values).float().to(device)
        y_train_torch = torch.from_numpy(y_train.values).float().to(device)
        used_data_torch = torch.from_numpy(used_data.values).float().to(device)
        
        # Training loop (simplified)
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model_instance(X_train_torch).squeeze()
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()
        
        # Predictions
        y_pred = model_instance(used_data_torch).detach().cpu().numpy()

    else:  # For sklearn models
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(used_data)

    # Evaluate the model's performance
    metrics = model_instance.evaluate(used_y, y_pred)

    return metrics, y_pred, used_y, used_data_full

def save_model_to_file(model, model_name, user_id, strategy_id, feature_set_name):
    """Save the model to a file and return the file path."""
    # Define directory and file name based on model details
    directory = "saved_models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{model_name}_{user_id}_{strategy_id}_{feature_set_name}.pkl")
    
    # Save the model
    if "PyTorch" in model_name:
        torch.save(model.state_dict(), file_path)
    else:
        joblib.dump(model, file_path)
    
    return file_path


def get_device_info(model_name):
    preferred_device = model_device_mapping.get(model_name, "CPU")
    return 'gpu' if preferred_device == "GPU" else 'cpu'
    
"""
def get_date_index(df, start_date, end_date):
    #Get indices corresponding to the date range in the DataFrame.
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    indices = df[mask].index.tolist()
    if not indices:
        raise ValueError(f"No data available between {start_date} and {end_date}")
    return min(indices), max(indices) + 1  # +1 to include the end date in slicing
"""

def start_training(model_name, hyperparams, df, top_k):
    ...
    for train_df, val_df in rolling_window_split(df):
        ...
        # Your training and validation logic here
        ...

    # After the loop ends, the last 2 years are your out-of-sample data.
    out_of_sample_start_date = df['date'].max() - pd.Timedelta(days=2*365)
    _, out_of_sample_start_idx = get_date_index(df, out_of_sample_start_date, df['date'].max())
    out_of_sample_data = df.iloc[out_of_sample_start_idx:]
    
    # You can now test on this out_of_sample_data
    ...

def get_hyperparameters_for_model(model_id):
    with db_manager.engine.connect() as connection:
        results = connection.execute(hyperparameter_configs.select().where(hyperparameter_configs.c.model_id == model_id)).fetchall()
    return [dict(row) for row in results]

def update_hyperparameters(model_id, updated_params):
    with db_manager.engine.connect() as connection:
        for param_name, value in updated_params.items():
            connection.execute(hyperparameter_configs.update().where(and_(
                hyperparameter_configs.c.model_id == model_id,
                hyperparameter_configs.c.hyperparam_name == param_name
            )).values(**value))

def decide_compute_resource(model_name):
    """
    Decide if the model should run on GPU or CPU.
    """
    preferred_device = model_device_mapping.get(model_name, "CPU")
    if preferred_device == "GPU" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def prepare_data_for_library(library, X, y, device='cpu'):
    """
    Convert data based on the library being used.
    """
    if library == "PyTorch":
        X = torch.from_numpy(X.values).float().to(device)
        y = torch.from_numpy(y.values).float().to(device)
    return X, y



def make_predictions(model_instance, library, X, device='cpu'):
    """
    Use the trained model to make predictions.
    """
    if library == "PyTorch":
        with torch.no_grad():
            model_instance.eval()  # Set the model to evaluation mode
            predictions = model_instance(X).cpu().numpy()

    else:  # Sklearn models
        predictions = model_instance.predict(X)

    return predictions

def prepare_data_for_model(X, y, model_name):
    library, _ = MODEL_MAPPER.get(model_name)
    
    if library == "PyTorch":
        X = torch.from_numpy(X.values).float()
        y = torch.from_numpy(y.values).float()
        # Additional reshaping if needed can be added here
    elif library == "Sklearn":
        # Convert y to 1D if it's 2D
        y = y.ravel()
    
    return X, y



def expanding_window_split(df, initial_train_size, window_size, test_size):
    # Split by symbols
    symbols = df['symbol'].unique()
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        
        # Define the limits
        test_start_idx = len(symbol_df) - test_size
        initial_train_end_idx = initial_train_size
        
        # Loop until we run out of data to create a new validation set
        while initial_train_end_idx < test_start_idx:
            yield (symbol_df.iloc[:initial_train_end_idx], symbol_df.iloc[initial_train_end_idx:initial_train_end_idx+window_size])
            initial_train_end_idx += window_size
        
        # Yield the final train and test split
        yield (symbol_df.iloc[:test_start_idx], symbol_df.iloc[test_start_idx:])



def get_date_index(df, start_date, end_date):
    """Get indices corresponding to the date range in the DataFrame."""
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    indices = df[mask].index.tolist()
    
    print(f"Indices for {start_date} to {end_date}: {indices}")
    
    if not indices:
        raise ValueError(f"No data available between {start_date} and {end_date}")
    
    return min(indices), max(indices) + 1  # +1 to include the end date in slicing





def get_data_for_phase(df, phase, validation_window_number=None):
    
    # Filter based on the phase
    if phase == "train":
        X_train = df[df['current_phase'] == 'train'].drop(columns=['target', 'current_phase'])
        y_train = df[df['current_phase'] == 'train']['target']
        X_train_full = df[df['current_phase'] == 'train']
        return X_train, y_train, X_train_full
    
    elif phase.startswith("validation"):
        if validation_window_number is None:
            raise ValueError("For validation phase, a validation_window_number must be provided.")
        X_validate = df[df['current_phase'] == f'validation_{validation_window_number}'].drop(columns=['target', 'current_phase'])
        y_validate = df[df['current_phase'] == f'validation_{validation_window_number}']['target']
        X_val_full = df[df['current_phase'] == f'validation_{validation_window_number}']
        return X_validate, y_validate, X_val_full
    
    elif phase == "test":
        X_test = df[df['current_phase'] == 'test'].drop(columns=['target', 'current_phase'])
        y_test = df[df['current_phase'] == 'test']['target']
        X_test_full = df[df['current_phase'] == 'test']
        return X_test, y_test, X_test_full
    
    else:
        raise ValueError(f"Invalid phase: {phase}")


def initiate_training(model_name, hyperparams):
    """
    Create an instance of the model to be trained.
    """
    library, ModelClass = MODEL_MAPPER.get(model_name)
    if not ModelClass:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_instance = ModelClass(**hyperparams)
    return model_instance, library


def fit_model(model_instance, library, X_train, y_train, device='cpu'):
    """
    Fit the model on the training data.
    """
    if library == "PyTorch":
        model_instance.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_instance.parameters())
        for epoch in range(10):  # Adjust as required
            optimizer.zero_grad()
            outputs = model_instance(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    else:  # Sklearn models
        model_instance.fit(X_train, y_train)

    return model_instance



def generate_model_filename(user_id, model_name, phase):
    """
    Generates a unique filename based on user_id, model_name, timestamp, and phase.
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Extract first letter from model_name and phase
    model_initial = model_name[0].upper() if model_name else "M" # Default to 'M' if model_name is empty
    phase_initial = phase[0].upper() if phase else "P" # Default to 'P' if phase is empty

    # Construct the filename
    filename = f"{user_id}_{model_initial}{phase_initial}_{timestamp}.pkl"

    return filename


def json_serial(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()
    raise TypeError("Type not serializable")


def check_items_type(obj):
    """Check and print the types of items in a list or dict."""
    if isinstance(obj, list):
        for item in obj:
            #print(f"List item: {item}, Type: {type(item)}")
            check_items_type(item)  # Recursive call if nested structure
    elif isinstance(obj, dict):
        for key, value in obj.items():
            #print(f"Dict key: {key}, Value: {value}, Type: {type(value)}")
            check_items_type(value)  # Recursive call if nested structure

def deep_convert_dict(d):
    """Recursively convert dictionary keys to strings and values to built-in data types."""
    if not isinstance(d, (dict, list)):
        if isinstance(d, np.int64):
            return int(d)
        elif isinstance(d, np.float64):
            return float(d)
        return d
    if isinstance(d, list):
        return [deep_convert_dict(v) for v in d]
    return {deep_convert_dict(k): deep_convert_dict(v) for k, v in d.items()}




def prepare_data(df, top_k):
    # Drop the rows with NaN values in the 'target' column
    df = df.dropna(subset=['target'])
    feature_df = convert_features_columns(df, top_k)
    # Get the feature columns
    non_feature_columns = ['ts_event', 'Datetime', 'date', 'ticker_symbol', 'target', 'open', 'high', 'low', 'close']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]

    # Prepare the data
    X = feature_df.values
    X_full = df.values
    y = df['target'].values
    date_time = df['date'].values

    # Calculate the indices for splitting the data
    train_index = int(len(X) * 0.6)
    validation_index = train_index + int(len(X) * 0.1)

    # Split the data into train, validation, and test sets
    X_train, X_validation, X_test = X[:train_index], X[train_index:validation_index], X[validation_index:]
    X_train_full, X_val_full, X_test_full = X_full[:train_index], X_full[train_index:validation_index], X_full[validation_index:]
    y_train, y_validation, y_test = y[:train_index], y[train_index:validation_index], y[validation_index:]
    date_time_train, date_time_validation, date_time_test = date_time[:train_index], date_time[train_index:validation_index], date_time[validation_index:]

    # Convert the arrays to DataFrames with column names
    X_train = pd.DataFrame(X_train, columns=top_k)
    X_validation = pd.DataFrame(X_validation, columns=top_k)
    X_test = pd.DataFrame(X_test, columns=top_k)

    # Convert the arrays to DataFrames with column names
    X_train_full = pd.DataFrame(X_train_full, columns=[df.columns])
    X_val_full = pd.DataFrame(X_val_full, columns=[df.columns])
    X_test_full = pd.DataFrame(X_test_full, columns=[df.columns])

    # For now, remove datetime
    # X_train['Datetime'] = date_time_train.ravel()
    # X_validation['Datetime'] = date_time_validation.ravel()
    # X_test['Datetime'] = date_time_test.ravel()

    y_train = pd.Series(y_train, name='target')
    y_validation = pd.Series(y_validation, name='target')
    y_test = pd.Series(y_test, name='target')
    logger.info(f"====== X_train: {X_train}")
    logger.info(f"====== X_validation: {X_validation}")
    logger.info(f"====== X_test: {X_test}")
    logger.info(f"====== y_train: {y_train}")
    logger.info(f"====== y_validation: {y_validation}")
    logger.info(f"====== y_test: {y_test}")
    return X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_full, X_val_full, X_test_full


def assign_phases_to_dataframe(df):
    # Determine the time series boundaries
    min_date = df['date'].min()
    max_date = df['date'].max()

    print(f"Min Date: {min_date}")
    print(f"Max Date: {max_date}")

    # Calculate total number of days across entire time series
    total_days = (max_date - min_date).days
    print(f"Total Days: {total_days}")

    # Calculate the split sizes in days
    initial_train_days = round(total_days * 0.50)
    validation_days = round(total_days * 0.10)  # Each validation period
    test_days = round(total_days * 0.20)

    # Determine the date boundaries for each phase
    train_end_date = min_date + pd.Timedelta(days=initial_train_days)
    validation_1_end_date = train_end_date + pd.Timedelta(days=validation_days)
    validation_2_end_date = validation_1_end_date + pd.Timedelta(days=validation_days)
    validation_3_end_date = validation_2_end_date + pd.Timedelta(days=validation_days)
    test_start_date = max_date - pd.Timedelta(days=test_days)

    def assign_phases_for_symbol(group):
        # Assign train
        group.loc[group['date'] < train_end_date, 'current_phase'] = 'train'

        # Assign validation splits
        group.loc[(group['date'] >= train_end_date) & (group['date'] < validation_1_end_date), 'current_phase'] = 'validation_1'
        group.loc[(group['date'] >= validation_1_end_date) & (group['date'] < validation_2_end_date), 'current_phase'] = 'validation_2'
        group.loc[(group['date'] >= validation_2_end_date) & (group['date'] < validation_3_end_date), 'current_phase'] = 'validation_3'

        # Assign test
        group.loc[group['date'] >= test_start_date, 'current_phase'] = 'test'

        return group

    # Group by symbol and apply the function
    df = df.groupby('symbol').apply(assign_phases_for_symbol)

    # Reset the index to remove the MultiIndex created by groupby
    df.reset_index(drop=True, inplace=True)

    return df, min_date, max_date



def convert_numpy_types(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


def check_and_convert(value):
    if isinstance(value, np.generic):
        return np.asscalar(value)
    elif isinstance(value, (np.ndarray,)):  # Convert arrays/lists
        return value.tolist()
    elif pd.isna(value):  # Check for NaN and fill with 0
        return 0
    return value
