from sqlalchemy import text
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, MetaData, Table, select
import pandas as pd
import boto3
import multiprocessing
import joblib
import traceback
import os
import sqlalchemy
from sqlalchemy import and_
#from modeling_classes_classification_v2 import SVMModel, DecisionTreeModel, ANNModel, LinearModel
from modeling_classes_classification_v3 import *
from database_manager import DatabaseManager
from Visualization_v3 import Visualization
# from train_handler import train_user_model, save_model_to_file, get_device_info, expanding_window_split, get_date_index, json_serial, check_items_type, deep_convert_dict, convert_numpy_types
#from constants import model_device_mapping, MODEL_MAPPER
# from utils import numpy_datetime_to_datetime, safely_get_value
from gpu_instance_handler import GpuInstanceHandler
from sqlalchemy import or_
import logging
from train_handler_v5 import *
import json
import numpy as np
from constants_v4 import *
from flask import Response
from datetime import date
from sqlalchemy import and_  # Import the `and_` function for compound WHERE clauses
from datetime import datetime
from sqlalchemy import func
from flask_cors import CORS


log_filename = 'logs/modeling_endpoint_api.log'

# Clear the log file
# with open(log_filename, 'w') as file:
#     pass

# Setup logging
#logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(message)s')
#logger = logging.getLogger()




# Clear the log file
with open(log_filename, 'w') as file:
    pass


logger = logging.getLogger('MyAppLogger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




# Connect to user_db
db_manager = DatabaseManager(
    username="postgres", 
    password="postgres", 
    host="localhost", 
    db_name="internaldb",
    # use_ssl = True

)

# Tables
feature_selection_scores = Table('feature_selection_scores', db_manager.meta, autoload_with=db_manager.engine)
model_selection = Table('model_selection', db_manager.meta, autoload_with=db_manager.engine)
# added feature_set_name, user_id, strategy_id columns to db manually through pgadmin
hyperparameters = Table('hyperparameters', db_manager.meta, autoload_with=db_manager.engine)
# added user_id column
evaluation_metrics = Table('evaluation_metrics', db_manager.meta, autoload_with=db_manager.engine)
#daily_stock_data = Table('daily_stock_data', db_manager.meta, autoload_with=db_manager.engine)
strategy_objective = Table('strategy_objective', db_manager.meta, autoload_with=db_manager.engine)
normalized_dataframe_table = Table('normalized_dataframe', db_manager.meta, autoload_with=db_manager.engine)
trained_models = Table('trained_models', db_manager.meta, autoload_with=db_manager.engine)  # Assuming this table exists
# changed user_id column from integer to text and made id as auto increment 
return_predictions = Table('return_predictions', db_manager.meta, autoload_with=db_manager.engine)  # Assuming this table exists
modeling_dataframe_table = Table('modeling_dataframe', db_manager.meta, autoload_with=db_manager.engine)
return_prediction_probability = Table('return_prediction_probability', db_manager.meta, autoload_with=db_manager.engine)
feature_sets = Table('feature_sets', db_manager.meta, autoload_with=db_manager.engine)
api_data = Table('api_data', db_manager.meta, autoload_with=db_manager.engine)



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


normalized_dataframe_table = Table('normalized_dataframe', db_manager.meta, autoload_with=db_manager.engine)
feature_selection_scores = Table('feature_selection_scores', db_manager.meta, autoload_with=db_manager.engine)

data_filename = 'training_data.txt'



class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        return super(EnhancedJSONEncoder, self).default(obj)


app.json_encoder = EnhancedJSONEncoder



@app.route('/model_training', methods=['POST'])
def model_training():
    try:
        data = request.get_json()
        print(f"API called with data: {data}")

        strategy_id = data['strategy_id']
        model_name = data['model_name']
        phase = data.get('phase', 'train')


        model_class_info = MODEL_MAPPER.get(model_name)
        if not model_class_info:
            error_message = f"Invalid model_name '{model_name}' received. Available models: {list(MODEL_MAPPER.keys())}"
            logger.error(error_message)
            # Return the updated error message in the response
            return jsonify({"error": error_message}), 400

        library, model_class = model_class_info
        print(f"Model class for {model_name}: {model_class.__name__}, Library: {library}")


        hyperparams = None

        # Retrieve hyperparameters from the request if phase is 'train'
        if phase == 'train':
            hyperparams = data.get('hyperparameters')

        if not hyperparams:
            # Query to retrieve hyperparameters for phases other than 'train'
            with db_manager.engine.connect() as connection:
                hyperparams_stmt = text("""
                    SELECT hyperparameters FROM evaluation_metrics 
                    WHERE strategy_id = :strategy_id AND phase = 'train' 
                    ORDER BY id DESC LIMIT 1
                """)
                hyperparams = connection.execute(hyperparams_stmt, {"strategy_id": strategy_id}).scalar()

                if not hyperparams:
                    return jsonify({"error": f"No hyperparameters found for the provided strategy_id during '{phase}' phase."}), 404
                else:
                    # Convert the hyperparameters string to a dictionary
                    try:
                        hyperparams = json.loads(hyperparams.replace("'", "\""))
                    except json.JSONDecodeError as e:
                        return jsonify({"error": f"Error decoding hyperparameters: {e}"}), 500

        print(f"Received hyperparameters: {hyperparams}")

        # Query to get the max version from feature_selection_scores
        with db_manager.engine.connect() as connection:
            max_version_stmt_feature_scores = text(f"SELECT MAX(version) FROM feature_selection_scores WHERE strategy_id = :strategy_id")
            max_version_feature_scores = connection.execute(max_version_stmt_feature_scores, {"strategy_id": strategy_id}).scalar() or 0

            # Modified Query: Select feature_name and their IDs using the max version
            query = text("""
                SELECT feature_name, id FROM feature_selection_scores 
                WHERE strategy_id = :strategy_id AND version = :max_version 
                AND (remove_feature != 't' OR remove_feature IS NULL)
                ORDER BY id
            """)
            feature_score_results = connection.execute(query, {"strategy_id": strategy_id, "max_version": max_version_feature_scores}).fetchall()

            # Sort features by their 'id'
            sorted_feature_info = sorted(feature_score_results, key=lambda x: x.id)
            unique_feature_names = [info.feature_name for info in sorted_feature_info]

            print("Sorted Unique Feature Names:")
            print(unique_feature_names)


            # Fetch the max version from feature_sets for the given strategy_id
            max_version_stmt_feature_sets = text("SELECT MAX(version) FROM feature_sets WHERE strategy_id = :strategy_id")
            max_version_feature_sets = connection.execute(max_version_stmt_feature_sets, {"strategy_id": strategy_id}).scalar() or 0

            # Fetch feature details from feature_sets using the max version fetched above
            feature_details_stmt = select(
                feature_sets.c.feature_name,
                feature_sets.c.feature_parameters
            ).where(
                and_(
                    feature_sets.c.strategy_id == strategy_id,
                    feature_sets.c.version == max_version_feature_sets
                )
            )
            feature_details_results = connection.execute(feature_details_stmt).fetchall()

            print("Feature details results:")
            print(feature_details_results)



        # Create a list of tuples (feature_name, feature_parameters) filtering out features not in unique_feature_names
        filtered_feature_details = [
            (row.feature_name, row.feature_parameters)
            for row in feature_details_results
            if row.feature_name in unique_feature_names
        ]

        print("Unique feature names expected:")
        print(unique_feature_names)

        # Directory to store the .pkl files
        directory = os.path.join(os.getcwd(), str(strategy_id))
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")

        # Define the .pkl file name
        temp_features_filename = f"{model_name}_strategy_{strategy_id}_features_v{max_version_feature_scores}.pkl"

        # Define the full path to the .pkl file
        full_pkl_file_path = os.path.join(directory, temp_features_filename)

        # Save feature names to the .pkl file
        try:
            with open(full_pkl_file_path, 'wb') as file:
                joblib.dump(unique_feature_names, file)
            print(f"Saved feature names and parameters to {full_pkl_file_path}")
        except Exception as e:
            print(f"Error saving data to {full_pkl_file_path}: {e}")
            raise


        # Query 2: Extracting data from normalized_dataframe_table based on strategy_id and max(version)
        with db_manager.engine.connect() as connection:
            max_version_stmt_normalized = text(f"SELECT MAX(version) FROM normalized_dataframe WHERE strategy_id = :strategy_id")
            max_version_normalized = connection.execute(max_version_stmt_normalized, {"strategy_id": strategy_id}).scalar() or 0

            df_results_stmt = select(normalized_dataframe_table).where(
                and_(
                    normalized_dataframe_table.c.strategy_id == strategy_id,
                    normalized_dataframe_table.c.version == max_version_normalized
                )
            )
            df_results = connection.execute(df_results_stmt).fetchall()


        normalized_df = pd.DataFrame(df_results)
        print("Original Dataframe:")
        print(normalized_df.head())
        print("Column names in the dataframe:", normalized_df.columns.tolist())  # Added print statement


        # Convert 'normalized_features' from stringified JSON to dictionary
        normalized_df['normalized_features'] = normalized_df['normalized_features'].str.replace('NaN', 'None').apply(eval)

        # Check for 'NaN' values in 'normalized_features'
        nan_rows = normalized_df[normalized_df['normalized_features'].astype(str).str.contains('NaN')]
        if not nan_rows.empty:
            print("Rows with 'NaN' in 'normalized_features':")
            print(nan_rows)

        # Expand the dictionary into separate columns
        features_df = normalized_df['normalized_features'].apply(pd.Series)

        print("\nExpanded Features:")
        print(features_df.head())

        # Keep only the columns that are present in unique_feature_names
        features_df = features_df[unique_feature_names]

        # Rename columns to match the database schema (feature1_, feature2_, etc.)
        #features_df.columns = [f'feature_{i+1}_' for i in range(len(features_df.columns))]

        # Print renamed feature columns
        print("\nRenamed Features DataFrame:")
        print(features_df.head())

        # Merge the expanded dataframe with the original dataframe
        normalized_df_selected_feature = pd.concat([normalized_df, features_df], axis=1)
        print("\nMerged Dataframe:")
        print(normalized_df_selected_feature.head())



        # Drop unnecessary columns
        normalized_df_selected_feature.drop(columns=['normalized_features', 'id'], inplace=True)
        print("\nAfter Dropping Unnecessary Columns:")
        print(normalized_df_selected_feature.head())

        # Drop rows with NA values
        normalized_df_selected_feature.dropna(inplace=True)

        # Sort the dataframe by 'date' and 'symbol'
        normalized_df_selected_feature.sort_values(by=['symbol', 'date'], inplace=True)

        # Set a default value for the 'current_phase' column
        normalized_df_selected_feature['current_phase'] = 'unassigned'

        # Assign phases to the dataframe
        normalized_df_selected_feature, _, _ = assign_phases_to_dataframe(normalized_df_selected_feature)


        # Instead of using phase_data, use the new logic
        if phase == 'train':
            train_data = normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'train']
        elif phase == 'validation_1':
            train_data = pd.concat([
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'train'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_1']
            ])
        elif phase == 'validation_2':
            train_data = pd.concat([
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'train'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_1'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_2']
            ])
        elif phase == 'validation_3':
            train_data = pd.concat([
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'train'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_1'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_2'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_3']
            ])
        elif phase == 'test':
            train_data = pd.concat([
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'train'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_1'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_2'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'validation_3'],
                normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'test']
            ])


        # Rename feature columns before bulk insertion
        feature_columns_present = [col for col in unique_feature_names if col in normalized_df_selected_feature.columns]
        new_feature_column_names = {old_name: f'feature{idx + 1}_' for idx, old_name in enumerate(feature_columns_present)}
        normalized_df_selected_feature.rename(columns=new_feature_column_names, inplace=True)

        # Rename feature columns in features_df to align with normalized_df_selected_feature
        features_df.rename(columns=new_feature_column_names, inplace=True)

        # Print to verify column names are aligned in features_df
        print("\nFeatures DataFrame After Renaming for Merge:")
        print(features_df.head())

        # Open a single connection for all database operations
        with db_manager.engine.connect() as connection:
            # Fetch the maximum version for the given strategy_id


            # Fetch the maximum version for the given strategy_id
            current_max_version = connection.execute(
                modeling_dataframe_table.select().where(
                    modeling_dataframe_table.c.strategy_id == strategy_id
                ).order_by(
                    modeling_dataframe_table.c.version.desc()
                ).limit(1).with_only_columns(modeling_dataframe_table.c.version)
            ).scalar()

            next_version = (current_max_version + 1) if current_max_version is not None else 1


            # Prepare data for bulk insertion
            to_insert_all_phases = []
            for _, row in normalized_df_selected_feature.iterrows():
                data_dict = row.to_dict()

                # Handle NaN values and other transformations
                for key, value in data_dict.items():
                    if isinstance(value, (int, float)) and np.isnan(value):
                        data_dict[key] = None

                # Include additional data like strategy_id, version, and date_inserted
                data_dict['strategy_id'] = strategy_id
                data_dict['version'] = next_version
                data_dict['date_inserted'] = datetime.now()
                data_dict['phase'] = data_dict['current_phase']

                # Add the renamed feature columns to the data_dict using the new column names
                for col in feature_columns_present:
                    renamed_col = new_feature_column_names[col]
                    data_dict[renamed_col] = row[renamed_col]  # Access the value using the renamed column name

                to_insert_all_phases.append(data_dict)



            # Print sample data to be inserted for verification
            print("\nSample Data Prepared for Bulk Insertion:")
            print(to_insert_all_phases[:5])

            logger.info("\nChecking 'target_return_threshold' in Sample Data:")
            for item in to_insert_all_phases[:5]:  # Changed from sample_to_insert to to_insert_all_phases
                if 'target_return_threshold' in item:
                    print(f"Symbol: {item.get('symbol', 'N/A')}, Date: {item.get('date', 'N/A')}, Target Return Threshold: {item['target_return_threshold']}")
                    logger.info(f"Symbol: {item.get('symbol', 'N/A')}, Date: {item.get('date', 'N/A')}, Target Return Threshold: {item['target_return_threshold']}")
                else:
                    logger.info(f"Symbol: {item.get('symbol', 'N/A')}, Date: {item.get('date', 'N/A')} - 'target_return_threshold' not found.")
                    print(f"Symbol: {item.get('symbol', 'N/A')}, Date: {item.get('date', 'N/A')} - 'target_return_threshold' not found.")


            # Execute the bulk insertion
            try:
                connection.execute(modeling_dataframe_table.insert(), to_insert_all_phases)
                connection.commit()
            except Exception as e:
                connection.rollback()
                print(f"Error during bulk insertion: {e}")
                logger.error(f"Error during bulk insertion: {e}")
                raise e

        # Rename features again before merging with normalized_df
        normalized_df_selected_feature.rename(columns=new_feature_column_names, inplace=True)

        # Print DataFrame after re-renaming to verify column names before merge
        print("\nDataFrame After Renaming Feature Columns for Merge:")
        print(normalized_df_selected_feature.head())


        # Now perform the merge using the renamed columns
        updated_normalized_df = normalized_df.merge(normalized_df_selected_feature[['current_phase', *new_feature_column_names.values()]], left_index=True, right_index=True, how='left')

        # Print the merged dataframe for verification
        print("\nUpdated Normalized DataFrame After Merge:")
        print(updated_normalized_df.head())


        print("\nFinal Dataframe with Phases:")
        print(normalized_df_selected_feature.head())

        unassigned_rows = normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == 'unassigned']
        if not unassigned_rows.empty:
            print("Rows that are unassigned:")
            print(unassigned_rows)
        else:
            print("All rows are assigned!")


        # Get a list of all unique phases with data
        unique_phases = normalized_df_selected_feature['current_phase'].unique().tolist()
        print("Unique phases:", unique_phases)  # Added print statement

        # Update the 'strategy_objective' table to insert the list of phases
        with db_manager.engine.connect() as connection:
            update_stmt = strategy_objective.update().where(
                strategy_objective.c.id == strategy_id
            ).values(phases=unique_phases)
            
            connection.execute(update_stmt)
            connection.commit()
            print("Updated 'strategy_objective' table with phases.")  # Added print statement


        # Rename features to original names before preparing X_train and y_train
        with open(full_pkl_file_path, 'rb') as f:
            feature_order = joblib.load(f)
            print("Feature order loaded from:", full_pkl_file_path)  # Added print statement

        original_feature_column_names = {f'feature{idx + 1}_': name for idx, name in enumerate(unique_feature_names)}
        normalized_df_selected_feature.rename(columns=original_feature_column_names, inplace=True)
        print("Renamed feature columns to original names.")  # Added print statement

        # Prepare X_train and y_train with original feature names
        train_data = normalized_df_selected_feature[normalized_df_selected_feature['current_phase'] == phase]
        X_train = train_data[unique_feature_names]  # Use features in the sequence specified in unique_feature_names
        y_train = train_data['target']
        print("Prepared X_train and y_train.")  # Added print statement

        metadata_df = train_data[['date', 'symbol', 'target', 'current_phase', 'target_return', 'target_return_threshold']]
        print("Metadata dataframe prepared.")  # Added print statement


        # Print Training Data for Inspection
        print("Training data (features):\n", X_train.head())
        print("Training data (target):\n", y_train.head())

        # Model Initialization and Training
        if model_name == 'SVMModel':
            model_class = SVMModel
        elif model_name == 'DecisionTreeModel':
            model_class = DecisionTreeModel
        elif model_name == 'LinearModel':
            model_class = LinearModel
        elif model_name == 'ANNModel':
            model_class = ANNModel
        elif model_name == 'RandomForestModel':
            model_class = RandomForestModel
        else:
            return {"message": f"Invalid model_name: {model_name}"}, 400

        print("model class")
        print(model_class)
        print(model_class.available_hyperparameters)


        # Before model initialization and training
        if not all(param in model_class.available_hyperparameters for param in hyperparams.keys()):
            return {"message": "Invalid hyperparameters provided"}, 400
        print(f"Initializing model: {model_name} with hyperparameters: {hyperparams}")        
        logger.info(f"Initializing model: {model_name} with hyperparameters: {hyperparams}")

        try:
            model = model_class(**hyperparams)
            logger.info(f"Model {model_name} initialized successfully")
            print(f"Model {model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            print(f"Error initializing model {model_name}: {e}")
            return jsonify({"error": f"Model initialization error: {e}"}), 500
        try:
            logger.info("Starting training for model: {}".format(model_name))            
            model.fit(X_train, y_train)
            logger.info(f"Model {model_name} trained successfully")
            temp_file_name = os.path.join(directory, f"temp_model_strategy_{strategy_id}.pkl")
            joblib.dump(model, temp_file_name)
            logger.info(f"Model {model_name} trained and saved successfully")
        except Exception as e:
            logger.error(f"Error during model training or saving: {e}")
            return jsonify({"error": f"Model training or saving error: {e}"}), 500

        logger.info(f"X_train type: {type(X_train)}, shape: {X_train.shape if X_train is not None else 'None'}")
        logger.info(f"y_train type: {type(y_train)}, shape: {y_train.shape if y_train is not None else 'None'}")

        # Train the Model and Handle Potential Errors
        temp_file_name = os.path.join(directory, f"temp_model_strategy_{strategy_id}.pkl")

        # Predict on the training data
        y_pred_proba = model.predict_proba(X_train)
        predicted_probabilities = y_pred_proba[:, 1]
        y_pred = (predicted_probabilities > 0.5).astype(int)
        print("Predicted Labels:", y_pred)

        # Construct a result DataFrame for predictions
        result_df = pd.concat([metadata_df.reset_index(drop=True), 
                              pd.Series(predicted_probabilities, name='predicted_probability')], axis=1)



        # Log the result DataFrame
        logger.info("Result DataFrame with Predicted Probabilities:\n{}".format(result_df.head(1000).to_string()))



        # Convert the result DataFrame into a dictionary
        predictions = result_df.to_dict(orient='records')


        # Evaluate the model using the Visualization class
        viz = Visualization(model)
        metrics = {}
        auc_roc_curve_data = viz.auc_roc_curve(y_train, predicted_probabilities)

        # Convert 'Infinity' threshold to a string to make it JSON serializable
        auc_roc_curve_data['thresholds'] = ["Infinity" if x == float('inf') else x for x in auc_roc_curve_data['thresholds']]
        metrics["AUC-ROC Curve"] = auc_roc_curve_data
        metrics["Confusion Matrix"] = viz.confusion_matrix(y_train, y_pred)
        metrics["Precision, Recall, Fscore"] = viz.precision_recall_fscore(y_train, y_pred)
        metrics["AUC-ROC"] = viz.auc_roc(y_train, y_pred)
        metrics["Log Loss"] = viz.log_loss(y_train, y_pred)
        metrics["Accuracy"] = viz.accuracy(y_train, y_pred)

        # Extract evaluation metrics and prepare them for insertion
        confusion = metrics["Confusion Matrix"]
        print(confusion)

        file_name = f"{model_name}_strategy_{strategy_id}_train.pkl"  # This will be overwritten later, but we need it to ensure your metrics data structure is consistent


        metrics_data = {
            'strategy_id': strategy_id,
            'phase': phase, 
            'true_positives': next(item['value'] for item in confusion if item["name"] == "True Positives"),
            'false_negatives': next(item['value'] for item in confusion if item["name"] == "False Negatives"),
            'false_positives': next(item['value'] for item in confusion if item["name"] == "False Positives"),
            'true_negatives': next(item['value'] for item in confusion if item["name"] == "True Negatives"),
            'precision': metrics["Precision, Recall, Fscore"][0]['value'],
            'recall': metrics["Precision, Recall, Fscore"][1]['value'],
            'fscore': metrics["Precision, Recall, Fscore"][2]['value'],
            'auc_roc': metrics["AUC-ROC"][0]['value'],  
            'log_loss': metrics["Log Loss"][0]['value'],  
            'accuracy': metrics["Accuracy"][0]['value'],
            'model_name': model_name,  # Add model name
            'hyperparameters': json.dumps(hyperparams),  # Convert hyperparameters to JSON string format
            'file_name': file_name,  # Add the file name
            'features_file': temp_features_filename, 
            'retrain': True if phase != 'train' else False

        }


        # Convert numpy types if any and fill NaN values with 0
        metrics_data = {k: 0 if pd.isna(v) else convert_numpy_types(v) for k, v in metrics_data.items()}

        with db_manager.engine.connect() as connection:
            # Inserting evaluation_metrics data

            latest_version = connection.execute(
                select(func.max(evaluation_metrics.c.version).label("max_version")).where(
                    (evaluation_metrics.c.strategy_id == strategy_id)  
                )
            ).scalar()


            # Set the new version based on the current max version
            next_version = latest_version + 1 if latest_version is not None else 1
            metrics_data['version'] = next_version
            
            # Set the current date and time for the date_inserted column
            metrics_data['date_inserted'] = datetime.now()

            # Execute the insertion
            result = connection.execute(evaluation_metrics.insert(), metrics_data)
            connection.commit()  # Committing within the context manager

            # Get the ID of the inserted row
            inserted_id = result.inserted_primary_key[0]

            # Construct a new filename including the fetched ID
            new_file_name = f"{model_name}_strategy_{strategy_id}_id_{inserted_id}.pkl"
            new_features_file_name = f"{model_name}_strategy_{strategy_id}_features_v{max_version_feature_scores}_id_{inserted_id}.pkl"

            # Full paths for renaming
            full_new_file_path = os.path.join(directory, new_file_name)
            full_new_features_file_path = os.path.join(directory, new_features_file_name)

            # Rename the features file
            os.rename(full_pkl_file_path, full_new_features_file_path)

            # Rename the model file
            os.rename(os.path.join(directory, f"temp_model_strategy_{strategy_id}.pkl"), full_new_file_path)

            # Update the file_name in the evaluation_metrics table with the new filename
            update_stmt = (
                evaluation_metrics.update()
                .where(evaluation_metrics.c.id == inserted_id)
                .values(file_name=new_file_name, features_file=new_features_file_name) # Also set the features_file
            )
            connection.execute(update_stmt)
            connection.commit()


            # Inserting data to return_prediction_probability table
            
            to_insert_all_groups = []
            for _, group in result_df.groupby(['current_phase']):
                # Fetch the latest version for the given 'phase' and 'strategy_id' combination


                latest_version = connection.execute(
                    select(func.max(return_prediction_probability.c.version)).where(
                        (return_prediction_probability.c.current_phase == group['current_phase'].iloc[0]) & 
                        (return_prediction_probability.c.strategy_id == strategy_id)
                    )
                ).scalar()

                next_version = latest_version + 1 if latest_version is not None else 1


                # Assign the next_version and evaluation_metrics_id to the entire group
                group['version'] = next_version
                group['evaluation_metrics_id'] = inserted_id
                group['strategy_id'] = strategy_id

                # Convert the DataFrame to a list of dictionaries and append to main list
                to_insert_all_groups.extend(group.to_dict(orient='records'))

            # Insert all data in one go
            connection.execute(return_prediction_probability.insert(), to_insert_all_groups)
            connection.commit()


        # Ensure JSON compatibility
        for key, value in metrics.items():
            check_items_type(value)
        

        # Define the function to check and convert values
        def check_and_convert(value):
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                return int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                return float(value) if not np.isnan(value) else 0
            elif isinstance(value, (np.ndarray,)): # This line added for arrays/lists
                return value.tolist()
            return value

        # Define the function to convert a deep dictionary
        def deep_convert_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = deep_convert_dict(value)
                else:
                    d[key] = check_and_convert(value)
            return d


        metrics = deep_convert_dict(metrics)
        print(metrics)

        def save_response_to_txt(response, filename):
            with open(filename, 'w') as f:
                json.dump(response, f, cls=EnhancedJSONEncoder, indent=4)


        # Convert 'predictions' which is a list of dictionaries
        for prediction in predictions:
            for key, value in prediction.items():
                prediction[key] = check_and_convert(value)


        # Create mock prediction data
        mock_predictions = [
            {"date": "2020-09-01 00:00:00", "symbol": "XOM", "target": 0.0, "current_phase": "train", "target_return": 0.0, "predicted_probability": 0.5},
            {"date": "2020-09-02 00:00:00", "symbol": "XOM", "target": 0.0, "current_phase": "train", "target_return": 0.0, "predicted_probability": 0.5}
            # Add more mock records as needed
        ]

        metrics = deep_convert_dict(metrics)

        # Initialize the Visualization class only for the line of best fit calculation in model predicted probabilty vs target return
        viz = Visualization(None)  # Assuming no model is needed for this summary

        # Generate the predict_probability_line_best_fit data
        predict_probability_line = viz.predict_probability_line_best_fit(result_df, num_bins=10)


        # Apply conversion to summary predictions if needed
        #predict_probability_line = deep_convert_dict(predict_probability_line)

        # Convert each dictionary in 'predict_probability_line'
        #converted_predict_probability_line = [deep_convert_dict(item) for item in predict_probability_line]


        # Step 1: Create the response data
        response_data = {
            'metrics': metrics,
            'predictions': mock_predictions,  # Include mock prediction data
            'summary_predictions': predict_probability_line
        }

        # Step 2: Serialize the response data to JSON
        serialized_response_data = json.dumps(response_data, cls=EnhancedJSONEncoder)

        try:
            # Step 3: Database operations
            # Create and execute the insert statement for api_data
            with db_manager.engine.connect() as connection:
                insert_stmt = api_data.insert().values(
                    strategy_id=strategy_id,
                    page='model_training',
                    api_response_data=serialized_response_data  # Use serialized JSON string
                )
                connection.execute(insert_stmt)
                connection.commit()

            # Step 4: Return the response
            # Create a Flask Response object with the JSON string
            response = Response(serialized_response_data, mimetype='application/json')
            return response

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


            # Save the response to a txt file
            save_response_to_txt(response, "api_response.txt")

            # Attempt to save the model using a try-except block
            try:
                joblib.dump(model, new_file_name)
            except Exception as e:
                # If model saving fails, log the error and continue
                print(f"Model saving error: {str(e)}")
                # You might want to also log this error to a file or database

            # Serialize the response data using json.dumps and the custom encoder
            response_json = json.dumps(response_data, cls=EnhancedJSONEncoder)

            # Return the serialized response data with the correct mimetype
            return Response(response_json, mimetype='application/json')

    except Exception as e:
        # If an error occurs, return the error message and traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500






@app.route('/model_prediction', methods=['POST'])
def model_prediction():

    print("Running")
    try:
        print("inside try block")
        # Logging the incoming request data
        data = request.get_json()
        logger.info(f"Received data for model prediction: {data}")

        strategy_id = data['strategy_id']
        model_name = data['model_name']
        current_phase = data['current_phase']
        phase = current_phase   

        logger.info(f"Extracted strategy_id: {strategy_id}, model_name: {model_name}, current_phase: {current_phase} from request")

        with db_manager.engine.connect() as connection:
            # Logging the execution of the hyperparameters query
            logger.info(f"Executing hyperparameters query for strategy_id {strategy_id}")
            hyperparams_stmt = text("""
                SELECT hyperparameters FROM evaluation_metrics 
                WHERE strategy_id = :strategy_id AND phase = 'train' 
                ORDER BY id DESC LIMIT 1
            """)
            hyperparams = connection.execute(hyperparams_stmt, {"strategy_id": strategy_id}).scalar()

            if not hyperparams:
                logger.error("No hyperparameters found for the provided strategy_id.")
                return jsonify({"error": "No hyperparameters found for the provided strategy_id."}), 404

            # Logging the retrieved hyperparameters
            logger.info(f"Retrieved hyperparameters: {hyperparams}")

            # Logging the execution of the max version query
            logger.info(f"Executing max version query for strategy_id {strategy_id} and current_phase {current_phase}")
            max_version_stmt_eval = text("""
                SELECT MAX(version) FROM evaluation_metrics 
                WHERE strategy_id = :strategy_id AND phase = :current_phase
            """)
            max_version_eval = connection.execute(max_version_stmt_eval, {"strategy_id": strategy_id, "current_phase": current_phase}).scalar() or 0
            next_version_eval = max_version_eval + 1
            logger.info(f"Retrieved max_version: {max_version_eval}, next_version: {next_version_eval}")

        # Extract the filename of the model
        with db_manager.engine.connect() as connection:
            logger.info(f"Fetching model filename for strategy_id {strategy_id}")
            file_name_stmt = text(f"SELECT file_name FROM evaluation_metrics WHERE strategy_id = :strategy_id ORDER BY id DESC LIMIT 1")
            model_filename = connection.execute(file_name_stmt, {"strategy_id": strategy_id}).scalar()

            if not model_filename:
                logger.error(f"No trained model found for strategy {strategy_id} using model {model_name} for current_phase {current_phase}.")
                return jsonify({"error": f"No trained model found for strategy {strategy_id} using model {model_name} for current_phase {current_phase}."}), 404

            logger.info(f"Model filename loaded from database: {model_filename}")

        # Construct the full path to the model file
        model_file_path = os.path.join(f'./{strategy_id}', model_filename)
        logger.info(f"Full path to the model file: {model_file_path}")


        # Load the saved model
        try:
            model = joblib.load(model_file_path)
            logger.info(f"Model loaded successfully from {model_file_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_file_path}")
            return jsonify({"error": f"No trained model file found at {model_file_path} for strategy {strategy_id} using model {model_name} for current_phase {current_phase}."}), 404

        # Retrieve the sorted feature names from feature_selection_scores
        with db_manager.engine.connect() as connection:
            max_version_stmt_feature_scores = text("SELECT MAX(version) FROM feature_selection_scores WHERE strategy_id = :strategy_id")
            max_version_feature_scores = connection.execute(max_version_stmt_feature_scores, {"strategy_id": strategy_id}).scalar() or 0

            feature_query = text("""
                SELECT feature_name, id FROM feature_selection_scores 
                WHERE strategy_id = :strategy_id AND version = :max_version 
                AND (remove_feature != 't' OR remove_feature IS NULL)
                ORDER BY id
            """)
            feature_results = connection.execute(feature_query, {"strategy_id": strategy_id, "max_version": max_version_feature_scores}).fetchall()
            unique_feature_names = [info.feature_name for info in sorted(feature_results, key=lambda x: x.id)]


        with db_manager.engine.connect() as connection:
            max_version_stmt = text(f"SELECT MAX(version) FROM modeling_dataframe WHERE strategy_id = :strategy_id AND current_phase = :current_phase")
            max_version = connection.execute(max_version_stmt, {"strategy_id": strategy_id, "current_phase": current_phase}).scalar() or 0

            df_results_stmt = select(modeling_dataframe_table).where(
                and_(
                    modeling_dataframe_table.c.strategy_id == strategy_id,
                    modeling_dataframe_table.c.current_phase == current_phase,
                    modeling_dataframe_table.c.version == max_version
                )
            )
            df_results = connection.execute(df_results_stmt).fetchall()

        # Convert SQL results to DataFrame
        phase_data = pd.DataFrame(df_results)
        print("Initial DataFrame structure:\n", phase_data.head())

        # Renaming columns in phase_data to match unique_feature_names
        new_column_names = {f'feature{i+1}_': name for i, name in enumerate(unique_feature_names)}
        phase_data.rename(columns=new_column_names, inplace=True)

        # Ensuring only columns in unique_feature_names are used for predictions
        X_phase = phase_data[unique_feature_names]

        # Check for any missing features
        missing_features = [f for f in unique_feature_names if f not in X_phase.columns]
        if missing_features:
            logger.info(f"Missing features in the data: {missing_features}")


        # Drop columns where all values are NaN
        phase_data = phase_data.dropna(axis=1, how='all')
        print(phase_data.dtypes)
        print("Columns after dropping NaN:", phase_data.columns)


        print("Columns in phase_data before accessing 'target':", phase_data.columns)

        # Extract the true target values
        y_true = phase_data['target']
        print("Extracted target values (y_true):\n", y_true.head())


        # Drop columns which aren't needed in the features
        #X_phase = phase_data.drop(columns=['id', 'strategy_id', 'symbol', 'date', 'target', 'current_phase', 'target_return', 'version', 'date_inserted'])
        
        # Drop rows with NaN values in the features, and align y_true and metadata_df with the cleaned X_phase
        valid_indices = X_phase.dropna().index
        X_phase = X_phase.loc[valid_indices]
        y_true = y_true[valid_indices]
        metadata_df = phase_data[['date', 'symbol', 'target', 'current_phase', 'target_return', 'target_return_threshold']].loc[valid_indices]

        logger.info("Code runs to this section before library_type model mapper")

        # Get prediction probabilities based on model type
        library_type, _ = MODEL_MAPPER[model_name]

        if library_type == "Sklearn":
            y_pred_proba = model.predict_proba(X_phase)
            predicted_probabilities = y_pred_proba[:, 1]
        elif library_type == "PyTorch":
            raise NotImplementedError("PyTorch predict_proba is not yet implemented.")
        else:
            raise ValueError(f"Unsupported model library: {library_type}")

        # Determine class labels based on threshold of 0.5
        y_pred = (predicted_probabilities > 0.5).astype(int)

        # Construct a result DataFrame
        result_df = pd.concat([metadata_df.reset_index(drop=True), 
                               pd.Series(predicted_probabilities, name='predicted_probability')], axis=1)


        # Evaluate the model using the Visualization class
        viz = Visualization(model)
        metrics = {}
        auc_roc_curve_data = viz.auc_roc_curve(y_true, predicted_probabilities)

        # Convert 'Infinity' threshold to a string to make it JSON serializable
        auc_roc_curve_data['thresholds'] = ["Infinity" if x == float('inf') else x for x in auc_roc_curve_data['thresholds']]
        metrics["AUC-ROC Curve"] = auc_roc_curve_data
        metrics["Confusion Matrix"] = viz.confusion_matrix(y_true, y_pred)
        metrics["Precision, Recall, Fscore"] = viz.precision_recall_fscore(y_true, y_pred)
        metrics["AUC-ROC"] = viz.auc_roc(y_true, y_pred)
        metrics["Log Loss"] = viz.log_loss(y_true, y_pred)
        metrics["Accuracy"] = viz.accuracy(y_true, y_pred)

        # Extract evaluation metrics and prepare them for insertion
        confusion = metrics["Confusion Matrix"]
        file_name = f"{model_name}_strategy_{strategy_id}_{current_phase}.pkl"


        metrics_data = {
            'strategy_id': strategy_id,
            'phase': current_phase,  # Use the provided phase instead of 'train'
            'true_positives': next(item['value'] for item in confusion if item["name"] == "True Positives"),
            'false_negatives': next(item['value'] for item in confusion if item["name"] == "False Negatives"),
            'false_positives': next(item['value'] for item in confusion if item["name"] == "False Positives"),
            'true_negatives': next(item['value'] for item in confusion if item["name"] == "True Negatives"),
            'precision': metrics["Precision, Recall, Fscore"][0]['value'],
            'recall': metrics["Precision, Recall, Fscore"][1]['value'],
            'fscore': metrics["Precision, Recall, Fscore"][2]['value'],
            'auc_roc': metrics["AUC-ROC"][0]['value'],
            'log_loss': metrics["Log Loss"][0]['value'],
            'accuracy': metrics["Accuracy"][0]['value'],
            'model_name': model_name,
            'file_name': model_filename,
            'version': next_version_eval,
            'hyperparameters': hyperparams,
        }

        def check_and_convert(value):
            if isinstance(value, np.generic):
                return value.item()
            return value

        # Convert the entire 'metrics_data' dictionary
        metrics_data = {k: check_and_convert(v) for k, v in metrics_data.items()}

        # Inserting into evaluation_metrics and retrieve the id
        with db_manager.engine.connect() as connection:
            result = connection.execute(evaluation_metrics.insert(), metrics_data)
            connection.commit()
            evaluation_metrics_id = result.inserted_primary_key[0]

            # Fetch the latest version for the 'strategy_id' and 'phase' combo
            latest_version = connection.execute(
                return_prediction_probability.select().where(
                    (return_prediction_probability.c.strategy_id == strategy_id) & 
                    (return_prediction_probability.c.phase == current_phase)
                ).order_by(
                    return_prediction_probability.c.version.desc()
                ).limit(1).with_only_columns(modeling_dataframe_table.c.version)

            ).scalar_one_or_none()

            # Determine the next version
            next_version = latest_version + 1 if latest_version is not None else 1

            # Prepare records for insertion into return_prediction_probability
            to_insert = []
            for _, row in result_df.iterrows():
                record = row.to_dict()
                record['strategy_id'] = strategy_id
                record['phase'] = current_phase
                record['version'] = next_version
                record['date_inserted'] = datetime.now()
                record['evaluation_metrics_id'] = evaluation_metrics_id
                to_insert.append(record)

            # Insert the records
            try:
                connection.execute(return_prediction_probability.insert(), to_insert)
                connection.commit()
            except Exception as e:
                connection.rollback()
                raise e

        print(f"Inserted {len(to_insert)} records into return_prediction_probability table.")

        # Convert metrics to serializable types
        for key, value in metrics.items():
            check_items_type(value)

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


        metrics = deep_convert_dict(metrics)

         # Function to save the response to a txt file
        #def save_response_to_txt(response, filename):
            #with open(filename, 'w') as f:
                #json.dump(response, f, cls=EnhancedJSONEncoder, indent=4)

        # Create mock prediction data
        mock_predictions = [
            {"date": "2020-09-01 00:00:00", "symbol": "XOM", "target": 0.0, "current_phase": "train", "target_return": 0.0, "predicted_probability": 0.5},
            {"date": "2020-09-02 00:00:00", "symbol": "XOM", "target": 0.0, "current_phase": "train", "target_return": 0.0, "predicted_probability": 0.5}
            # Add more mock records as needed
        ]

        # Initialize the Visualization class only for the line of best fit calculation in model predicted probabilty vs target return
        viz = Visualization(None)  # Assuming no model is needed for this summary

        # Generate the predict_probability_line_best_fit data
        predict_probability_line = viz.predict_probability_line_best_fit(result_df, num_bins=10)

        # Step 1: Create the response data
        response_data = {
            'metrics': metrics,
            'predictions': mock_predictions,  # Include mock prediction data
            'summary_predictions': predict_probability_line
        }

        # Step 2: Serialize the response data to JSON
        serialized_response_data = json.dumps(response_data, cls=EnhancedJSONEncoder)

        try:
            # Step 3: Database operations
            with db_manager.engine.connect() as connection:
                # Create and execute the insert statement for api_data
                insert_stmt = api_data.insert().values(
                    strategy_id=strategy_id,
                    page=current_phase,  # Use 'current_phase' for the 'page' field
                    api_response_data=serialized_response_data
                )
                connection.execute(insert_stmt)
                connection.commit()

            # Step 4: Return the response
            response = Response(serialized_response_data, mimetype='application/json')
            return response

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

            # Serialize the response data using json.dumps and the custom encoder
            response_json = json.dumps(response_data, cls=EnhancedJSONEncoder)

            # Return the serialized response data with the correct mimetype
            return Response(response_json, mimetype='application/json')

    except Exception as e:
        # If an error occurs, return the error message and traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/last_trained_model', methods=['GET'])
def last_trained_model():
    try:
        strategy_id = request.args.get('strategy_id')
        if not strategy_id:
            return jsonify({'error': 'strategy_id is required'}), 400

        with db_manager.engine.connect() as connection:
            # Query to find the maximum version for the given strategy_id
            max_version_stmt = text("SELECT MAX(version) FROM evaluation_metrics WHERE strategy_id = :strategy_id")
            max_version = connection.execute(max_version_stmt, {"strategy_id": strategy_id}).scalar() or 0

            # Query to find the model name of the last trained model
            model_name_stmt = select(evaluation_metrics.c.model_name).where(
                evaluation_metrics.c.strategy_id == strategy_id,
                evaluation_metrics.c.version == max_version
            )
            model_name_result = connection.execute(model_name_stmt).fetchone()

            if model_name_result:
                return jsonify({'model_name': model_name_result[0]})
            else:
                return jsonify({'error': 'Model not found for the given strategy_id'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5013, debug=False)  # Set host to '0.0.0.0' to make it accessible externally


