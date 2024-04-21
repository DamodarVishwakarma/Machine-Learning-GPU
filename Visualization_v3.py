import matplotlib.pyplot as plt
# import seaborn as sns
import base64
from io import BytesIO
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, explained_variance_score
import traceback
from sklearn.metrics import roc_curve
import numpy as np


class Visualization:
    def __init__(self, model):
        self.model = model


    def confusion_matrix(self, y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred)
        data = [{'name': 'True Positives', 'value': matrix[0][0]},
                {'name': 'False Negatives', 'value': matrix[0][1]},
                {'name': 'False Positives', 'value': matrix[1][0]},
                {'name': 'True Negatives', 'value': matrix[1][1]}]
        return data

    def precision_recall_fscore(self, y_test, y_pred):
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        return [{'name': 'Precision', 'value': precision}, {'name': 'Recall', 'value': recall}, {'name': 'Fscore', 'value': fscore}]

    def auc_roc(self, y_test, y_pred):
        auc = roc_auc_score(y_test, y_pred)
        return [{'name': 'AUC-ROC', 'value': auc}]

    def log_loss(self, y_test, y_pred):
        loss = log_loss(y_test, y_pred)
        return [{'name': 'Log Loss', 'value': loss}]


    def auc_roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Limit the number of data points
        max_points = 10  # Example: only keep 10 points
        indices = np.linspace(0, len(thresholds) - 1, max_points).astype(int)
        
        limited_fpr = fpr[indices]
        limited_tpr = tpr[indices]
        limited_thresholds = thresholds[indices]

        return {
            'fpr': limited_fpr.tolist(),
            'tpr': limited_tpr.tolist(),
            'thresholds': limited_thresholds.tolist()
        }



    def regression_metrics(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        return [{'name': 'MAE', 'value': mae}, {'name': 'MSE', 'value': mse}, {'name': 'RMSE', 'value': rmse}, {'name': 'R2 Score', 'value': r2}]

    def loss_per_epoch(self, history):
        return [{'epoch': i+1, 'loss': x} for i, x in enumerate(history)]


    def accuracy(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        return [{'name': 'Accuracy', 'value': acc}]

    def explained_variance(self, y_test, y_pred):
        evs = explained_variance_score(y_test, y_pred)
        return [{'name': 'Explained Variance Score', 'value': evs}]

    def residual_plots(self, y_test, y_pred):
        residuals = y_test - y_pred
        data = [{'actual': yt, 'predicted': yp, 'residual': r} for yt, yp, r in zip(y_test, y_pred, residuals)]
        return data

    def predict_probability_line_best_fit(self, df, num_bins=100):
        # Define the bin edges based on the range of predicted probabilities
        bin_edges = np.linspace(df['predicted_probability'].min(), df['predicted_probability'].max(), num_bins + 1)
        # Calculate bin centers
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Assign each probability to a bin and adjust for edge case
        df['probability_bin'] = np.digitize(df['predicted_probability'], bins=bin_edges, right=False)
        df['probability_bin'] = df['probability_bin'].apply(lambda x: min(x, num_bins) - 1) 

        # Calculate the average target return for each bin
        summary_df = df.groupby('probability_bin')['target_return'].mean().reset_index()

        # Remove bins with no values (NaNs)
        summary_df = summary_df.dropna()

        summary_df['bin_center'] = summary_df['probability_bin'].apply(lambda x: bin_centers[x])

        # Return a list of dictionaries with bin_center and average target_return
        return summary_df[['bin_center', 'target_return']].to_dict(orient='records')
