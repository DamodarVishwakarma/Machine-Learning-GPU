import torch
import torch.nn as nn
import torch.optim as optim


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out




class TFTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(TFTClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out)
        out = self.fc(transformer_out[:, -1, :])
        out = self.sigmoid(out)
        return out


class GBDTClassifier:
    def __init__(self, input_size, learning_rate, num_trees, max_depth):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def train(self, X, y, num_epochs, batch_size):
        for _ in range(self.num_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y)
            y_pred = tree.predict(X)
            y = y - self.learning_rate * y_pred
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return (y_pred > 0.5).astype(int)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_size):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x