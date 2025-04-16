import torch.nn as nn
import torch

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)  # 32 filters are applied
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 64 features
        self.relu = nn.ReLU() # 0 for negative values
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)  # CNN output added to lstm
        self.dropout = nn.Dropout(0.5)  # removes 50% neurons to avoid overfitting
        self.fc = nn.Linear(128, 1)  # converting 128 features to a single value

    def forward(self, x):
        x = x.unsqueeze(2)  
        x = self.relu(self.conv1(x))  # adding non-linearity to 1st layer
        x = self.relu(self.conv2(x))  
        x = x.permute(0, 2, 1)  # output ready from cnn foor lstm 
        x, _ = self.lstm(x)  
        x = self.dropout(x[:, -1, :])  # regularization to reduce overfitting
        x = self.fc(x)  
        return x.squeeze()  