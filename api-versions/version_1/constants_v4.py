from version_1.modeling_classes_classification_v3 import *
from version_2.gpu_model_classifications import *
import logging

log_filename = 'logs/modeling_endpoint_constants.log'

logger = logging.getLogger('MyAppLogger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Modify the model_device_mapping to support the PyTorch models
model_device_mapping = {
    "Sklearn_ANN": "CPU",
    "Sklearn_SVM": "CPU",
    "Sklearn_DecisionTree": "CPU",
    "Sklearn_LinearModel": "CPU",
    "Sklearn_RandomForestModel": "CPU",
    "PyTorch_LSTMModel": "GPU",
    "PyTorch_TFTModel": "GPU",
    "PyTorch_GBDTModel": "GPU",
    "PyTorch_TransformerModel": "GPU"
}

# Model Mapper now maps to a tuple containing the library and the model class
MODEL_MAPPER = {
    "SVMModel": ("Sklearn", SVMModel),
    "DecisionTreeModel": ("Sklearn", DecisionTreeModel),
    "LinearModel": ("Sklearn", LinearModel),
    "Sklearn_ANN": ("Sklearn", ANNModel),
    "RandomForestModel": ("Sklearn", RandomForestModel),
    "PyTorch_LSTMModel": ("PyTorch", LSTMModel),
    "PyTorch_TFTModel": ("PyTorch", TFTModel),
    "PyTorch_GBDTModel": ("PyTorch", GBDTModel),
    "PyTorch_TransformerModel": ("PyTorch", TransformerModel)
}

logger.info(f"MODEL_MAPPER: {MODEL_MAPPER}")