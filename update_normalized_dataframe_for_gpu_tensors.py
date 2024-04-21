from sqlalchemy import text, Table
from sqlalchemy.orm import Session
import torch
import numpy as np
from torch_geometric.data import Data

#### store the GPU torches after feature_selection/normalization
CREATE TABLE preprocessed_data (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER,
    version INTEGER,
    data BYTEA,
    date_inserted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


import io
import psycopg2
import torch

def store_preprocessed_data(strategy_id, version, preprocessed_data, db_connection):
    # Convert the preprocessed tensor data to bytes
    buffer = io.BytesIO()
    torch.save(preprocessed_data, buffer)
    data_bytes = buffer.getvalue()

    # Insert the preprocessed data into the database
    cursor = db_connection.cursor()
    query = "INSERT INTO preprocessed_data (strategy_id, version, data) VALUES (%s, %s, %s)"
    cursor.execute(query, (strategy_id, version, psycopg2.Binary(data_bytes)))
    db_connection.commit()
    cursor.close()

def load_preprocessed_data(strategy_id, version, db_connection):
    # Retrieve the preprocessed data from the database
    cursor = db_connection.cursor()
    query = "SELECT data FROM preprocessed_data WHERE strategy_id = %s AND version = %s"
    cursor.execute(query, (strategy_id, version))
    result = cursor.fetchone()
    cursor.close()

    if result is None:
        return None

    # Convert the retrieved bytes back to a tensor
    data_bytes = result[0]
    buffer = io.BytesIO(data_bytes)
    preprocessed_data = torch.load(buffer)

    return preprocessed_data






def extract_normalized_data(db_manager, strategy_id, version):
    session = Session(bind=db_manager.engine)
    
    try:
        normalized_dataframe_table = Table('normalized_dataframe', db_manager.meta, autoload_with=db_manager.engine)
        
        # Retrieve data from normalized_dataframe table based on strategy_id and version
        stmt = text(f"SELECT * FROM normalized_dataframe WHERE strategy_id = '{strategy_id}' AND version = {version}")
        result = session.execute(stmt)
        
        # Convert results to list of dictionaries
        normalized_data = [{column: value for column, value in zip(result.keys(), row)} for row in result]
        
        return normalized_data
    
    finally:
        session.close()

def preprocess_data_tensors(normalized_data):
    # Convert normalized data to PyTorch tensors
    features = [data['normalized_features'] for data in normalized_data]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    return features_tensor

def preprocess_data_encoder(normalized_data, encoder_model):
    # Extract features from normalized data
    features = [data['normalized_features'] for data in normalized_data]
    
    # Preprocess data using the selected encoder model
    preprocessed_data = encoder_model.transform(features)
    
    return preprocessed_data

def preprocess_data_graph(normalized_data):
    # Extract features and target from normalized data
    features = [data['normalized_features'] for data in normalized_data]
    target = [data['target'] for data in normalized_data]
    
    # Create edge index based on temporal or feature relationships
    num_samples = len(features)
    edge_index = torch.tensor([[i, i+1] for i in range(num_samples-1)] + 
                              [[i+1, i] for i in range(num_samples-1)], dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_index.t().contiguous(),
                y=torch.tensor(target, dtype=torch.float32))
    
    return data

# Usage example
strategy_id = ...
version = ...
user_selection = ...  # 'tensors', 'encoder', or 'graph'

# Extract normalized data from the database
normalized_data = extract_normalized_data(db_manager, strategy_id, version)

# Preprocess the data based on user's selection
if user_selection == 'tensors':
    preprocessed_data = preprocess_data_tensors(normalized_data)
elif user_selection == 'encoder':
    encoder_model = ...  # Instantiate the selected encoder model
    preprocessed_data = preprocess_data_encoder(normalized_data, encoder_model)
elif user_selection == 'graph':
    preprocessed_data = preprocess_data_graph(normalized_data)
else:
    raise ValueError("Invalid user selection. Choose from 'tensors', 'encoder', or 'graph'.")

# Use the preprocessed data for further processing or modeling