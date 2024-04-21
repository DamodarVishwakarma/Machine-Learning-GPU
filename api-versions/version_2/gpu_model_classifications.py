import torch
import torch.nn as nn
import torch.optim as optim
from version_2.gpu_model_architectures import LSTMClassifier, TFTClassifier, GBDTClassifier, TransformerClassifier
class BaseModel:
    def __init__(self):
        pass

class LSTMModel(BaseModel):
    available_hyperparameters = {
        'hidden_size': {
            'default': 128,
            'suggested': [64, 128, 256],
            'type': 'scaler'
        },
        'num_layers': {
            'default': 2,
            'suggested': [1, 2, 3],
            'type': 'scaler'
        },
        'learning_rate': {
            'default': 0.001,
            'suggested': [0.01, 0.001, 0.0001],
            'type': 'scaler'
        }
    }

    def __init__(self, input_size, hidden_size=None, num_layers=None, learning_rate=None):
        super().__init__()
        self.hyperparameters = {
            'hidden_size': hidden_size or self.available_hyperparameters['hidden_size']['default'],
            'num_layers': num_layers or self.available_hyperparameters['num_layers']['default'],
            'learning_rate': learning_rate or self.available_hyperparameters['learning_rate']['default']
        }
        self.model = LSTMClassifier(input_size, self.hyperparameters['hidden_size'], self.hyperparameters['num_layers'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
        self.criterion = nn.BCELoss()

    def fit(self, X, y, num_epochs=10, batch_size=32):
        self.model.train()
        for epoch in range(num_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().numpy()
        return predictions



class TFTModel(BaseModel):
    available_hyperparameters = {
        'hidden_size': {
            'default': 128,
            'suggested': [64, 128, 256],
            'type': 'scaler'
        },
        'num_layers': {
            'default': 2,
            'suggested': [1, 2, 3],
            'type': 'scaler'
        },
        'num_heads': {
            'default': 4,
            'suggested': [2, 4, 8],
            'type': 'scaler'
        },
        'learning_rate': {
            'default': 0.001,
            'suggested': [0.01, 0.001, 0.0001],
            'type': 'scaler'
        }
    }

    def __init__(self, input_size, hidden_size=None, num_layers=None, num_heads=None, learning_rate=None):
        super().__init__()
        self.hyperparameters = {
            'hidden_size': hidden_size or self.available_hyperparameters['hidden_size']['default'],
            'num_layers': num_layers or self.available_hyperparameters['num_layers']['default'],
            'num_heads': num_heads or self.available_hyperparameters['num_heads']['default'],
            'learning_rate': learning_rate or self.available_hyperparameters['learning_rate']['default']
        }
        self.model = TFTClassifier(input_size, self.hyperparameters['hidden_size'], self.hyperparameters['num_layers'], self.hyperparameters['num_heads'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
        self.criterion = nn.BCELoss()

    def fit(self, X, y, num_epochs=10, batch_size=32):
        self.model.train()
        for epoch in range(num_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().numpy()
        return predictions

class GBDTModel(BaseModel):
    available_hyperparameters = {
        'learning_rate': {
            'default': 0.1,
            'suggested': [0.01, 0.1, 0.3],
            'type': 'scaler'
        },
        'num_trees': {
            'default': 100,
            'suggested': [50, 100, 200],
            'type': 'scaler'
        },
        'max_depth': {
            'default': 3,
            'suggested': [3, 5, 7],
            'type': 'scaler'
        }
    }

    def __init__(self, input_size, learning_rate=None, num_trees=None, max_depth=None):
        super().__init__()
        self.hyperparameters = {
            'learning_rate': learning_rate or self.available_hyperparameters['learning_rate']['default'],
            'num_trees': num_trees or self.available_hyperparameters['num_trees']['default'],
            'max_depth': max_depth or self.available_hyperparameters['max_depth']['default']
        }
        self.model = GBDTClassifier(input_size, self.hyperparameters['learning_rate'], self.hyperparameters['num_trees'], self.hyperparameters['max_depth'])

    def fit(self, X, y, num_epochs=10, batch_size=32):
        self.model.train(X, y, num_epochs, batch_size)

    def predict(self, X):
        return self.model.predict(X)

class TransformerModel(BaseModel):
    available_hyperparameters = {
        'num_layers': {
            'default': 2,
            'suggested': [1, 2, 3],
            'type': 'scaler'
        },
        'num_heads': {
            'default': 4,
            'suggested': [2, 4, 8],
            'type': 'scaler'
        },
        'hidden_size': {
            'default': 128,
            'suggested': [64, 128, 256],
            'type': 'scaler'
        },
        'learning_rate': {
            'default': 0.001,
            'suggested': [0.01, 0.001, 0.0001],
            'type': 'scaler'
        }
    }

    def __init__(self, input_size, num_layers=None, num_heads=None, hidden_size=None, learning_rate=None):
        super().__init__()
        self.hyperparameters = {
            'num_layers': num_layers or self.available_hyperparameters['num_layers']['default'],
            'num_heads': num_heads or self.available_hyperparameters['num_heads']['default'],
            'hidden_size': hidden_size or self.available_hyperparameters['hidden_size']['default'],
            'learning_rate': learning_rate or self.available_hyperparameters['learning_rate']['default']
        }
        self.model = TransformerClassifier(input_size, self.hyperparameters['num_layers'], self.hyperparameters['num_heads'], self.hyperparameters['hidden_size'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
        self.criterion = nn.BCELoss()

    def fit(self, X, y, num_epochs=10, batch_size=32):
        self.model.train()
        for epoch in range(num_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().numpy()
        return predictions