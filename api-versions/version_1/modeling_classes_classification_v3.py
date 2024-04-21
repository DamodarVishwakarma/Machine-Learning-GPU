from sqlalchemy import text
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from abc import ABC, abstractmethod
import sqlalchemy
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import logging
import shap
import os

log_filename = 'logs/modeling_classifiacation_class.log'

log_directory = os.path.dirname(log_filename)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logger = logging.getLogger('MyAppLogger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)







# Base class for all models
class BaseModel(ABC):
    def __init__(self):
        self.hyperparameters = {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def calculate_feature_importance(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)
        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "AUC-ROC": auc_roc}

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except AttributeError:
            raise NotImplementedError(f"{type(self.model).__name__} does not have a predict_proba method.")


class SVMModel(BaseModel):
    available_hyperparameters = {
        'kernel': {
            'default': 'rbf',
            'suggested': ['linear', 'poly', 'rbf', 'sigmoid'],
            'type': 'dropdown'
        },
        'degree': {
            'default': 3,
            'suggested': [1, 2, 3, 4, 5],
            'type': 'scaler'
        },
        'gamma': {
            'default': 'scale',
            'suggested': ['scale', 'auto'],
            'type': 'dropdown'
        },
        'C': {
            'default': 1,
            'suggested': [1, 2, 3, 4, 5],  # Linear scale
            'type': 'scaler'
        }
    }

    def __init__(self, kernel=None, degree=None, gamma=None, C=None):
        super().__init__()
        self.hyperparameters = {
            'kernel': kernel or self.available_hyperparameters['kernel']['default'],
            'degree': degree or self.available_hyperparameters['degree']['default'],
            'gamma': gamma or self.available_hyperparameters['gamma']['default'],
            'C': C or self.available_hyperparameters['C']['default']
        }
        self.model = SVC(probability=True, **self.hyperparameters)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


    def calculate_feature_importance(self, X):
        try:
            # Create a SHAP explainer for the SVM model
            explainer = shap.KernelExplainer(self.model.predict_proba, X)

            # Calculate SHAP values for the input features
            shap_values = explainer.shap_values(X, nsamples=100)

            # Convert SHAP values to feature importance scores
            feature_importance = dict(zip(X.columns, abs(shap_values).mean(axis=0)))

            logger.info("Feature importance calculated successfully for SVMModel.")
            return feature_importance

        except Exception as e:
            logger.error(f"Error in calculating feature importance for SVMModel: {e}")
            raise






class DecisionTreeModel(BaseModel):
    available_hyperparameters = {
       'max_depth': {
            'default': None,
            'suggested': [1, 2, 3, 4, 5],
            'type': 'scaler'
        },
        'min_samples_split': {
            'default': 2,
            'suggested': [2, 3, 4, 5, 6],
            'type': 'scaler'
        },
        'min_samples_leaf': {
            'default': 1,
            'suggested': [1, 2, 3, 4, 5],
            'type': 'scaler'
        }
    }

    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        super().__init__()
        self.hyperparameters = {
            'max_depth': max_depth or self.available_hyperparameters['max_depth']['default'],
            'min_samples_split': min_samples_split or self.available_hyperparameters['min_samples_split']['default'],
            'min_samples_leaf': min_samples_leaf or self.available_hyperparameters['min_samples_leaf']['default']
        }
        self.model = DecisionTreeClassifier(**self.hyperparameters)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


    def calculate_feature_importance(self, X):
        try:
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            logger.info("Feature importance calculated successfully for DecisionTreeModel.")
            return feature_importance
        except Exception as e:
            logger.error(f"Error in calculating feature importance for DecisionTreeModel: {e}")
            raise



class RandomForestModel(BaseModel):
    available_hyperparameters = {
        'n_estimators': {
            'default': 90,
            'suggested': [10, 30, 50, 70, 90],  # Linear scale
            'type': 'scaler'
        },
        'max_depth': {
            'default': None,
            'suggested': [10, 20, 30, 40],
            'type': 'scaler'
        },
        'min_samples_split': {
            'default': 2,
            'suggested': [2, 4, 6, 8],  # Linear scale
            'type': 'scaler'
        },
        'min_samples_leaf': {
            'default': 1,
            'suggested': [1, 2, 3, 4],
            'type': 'scaler'
        }
    }

    def __init__(self, n_estimators=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        super().__init__()
        self.hyperparameters = {
            'n_estimators': n_estimators or self.available_hyperparameters['n_estimators']['default'],
            'max_depth': max_depth or self.available_hyperparameters['max_depth']['default'],
            'min_samples_split': min_samples_split or self.available_hyperparameters['min_samples_split']['default'],
            'min_samples_leaf': min_samples_leaf or self.available_hyperparameters['min_samples_leaf']['default']
        }
        self.model = RandomForestClassifier(**self.hyperparameters)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def calculate_feature_importance(self, X):
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            feature_importance = dict(zip(X.columns, abs(shap_values).mean(axis=0)))
            logger.info("Feature importance calculated successfully for RandomForestModel.")
            return feature_importance
        except Exception as e:
            logger.error(f"Error in calculating feature importance for RandomForestModel: {e}")
            raise


class LinearModel(BaseModel):
    # Define available hyperparameters with sensible defaults and suggestions
    available_hyperparameters = {
        'penalty': {
            'default': 'l2',
            'suggested': ['l1', 'l2'],
            'type': 'dropdown'
        },
        'C': {
            'default': 1,
            'suggested': [1, 2, 3, 4, 5],
            'type': 'scaler'
        },
        'solver': {
            'default': 'liblinear',  # Suitable for binary classification
            'suggested': ['liblinear', 'lbfgs'],  # Only including compatible solvers
            'type': 'dropdown'
        }
    }

    def __init__(self, penalty='l2', C=1, solver='liblinear'):
        super().__init__()
        # Assigning hyperparameters from the input or using defaults
        self.hyperparameters = {
            'penalty': penalty,
            'C': C,
            'solver': solver
        }

        # Validate and Initialize the Model
        try:
            self._validate_hyperparameters()
            self.model = LogisticRegression(**self.hyperparameters)
            logger.info("LinearModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error during LinearModel initialization: {e}")
            raise

    def _validate_hyperparameters(self):
        # Validation logic
        if self.hyperparameters['penalty'] not in self.available_hyperparameters['penalty']['suggested']:
            raise ValueError(f"Invalid penalty: {self.hyperparameters['penalty']}")
        if self.hyperparameters['C'] not in self.available_hyperparameters['C']['suggested']:
            raise ValueError(f"Invalid C value: {self.hyperparameters['C']}")
        if self.hyperparameters['solver'] not in self.available_hyperparameters['solver']['suggested']:
            raise ValueError(f"Invalid solver: {self.hyperparameters['solver']}")


    def fit(self, X, y):
        try:
            self.model.fit(X, y)
            logger.info("LinearModel fitted successfully.")
        except Exception as e:
            logger.error(f"Error in fitting LinearModel: {e}")
            raise

    def predict(self, X):
        return self.model.predict(X)


    def calculate_feature_importance(self, X):
        try:
            explainer = shap.LinearExplainer(self.model, X)
            shap_values = explainer.shap_values(X)
            feature_importance = dict(zip(X.columns, abs(shap_values).mean(axis=0)))
            logger.info("Feature importance calculated successfully for LinearModel.")
            return feature_importance
        except Exception as e:
            logger.error(f"Error in calculating feature importance for LinearModel: {e}")
            raise




class ANNModel(BaseModel):
    available_hyperparameters = {
        'hidden_layer_sizes': {
            'default': (100,),
            'suggested': [(50,), (100,), (50, 50), (100, 100)],
            'type': 'dropdown'
        },
        'activation': {
            'default': 'relu',
            'suggested': ['identity', 'logistic', 'tanh', 'relu'],
            'type': 'dropdown'
        },
        'solver': {
            'default': 'adam',
            'suggested': ['lbfgs', 'sgd', 'adam'],
            'type': 'dropdown'
        },
        'alpha': {
            'default': 0.0001,
            'suggested': [0.0001, 0.1],
            'type': 'scaler'
        }
    }

    def __init__(self, hidden_layer_sizes=None, activation=None, solver=None, alpha=None):
        super().__init__()
        self.hyperparameters = {
            'hidden_layer_sizes': hidden_layer_sizes or self.available_hyperparameters['hidden_layer_sizes']['default'],
            'activation': activation or self.available_hyperparameters['activation']['default'],
            'solver': solver or self.available_hyperparameters['solver']['default'],
            'alpha': alpha or self.available_hyperparameters['alpha']['default']
        }
        self.model = MLPClassifier(**self.hyperparameters)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)