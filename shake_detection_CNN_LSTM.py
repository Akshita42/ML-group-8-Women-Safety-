import torch.nn as nn # pytorch library for deep learning models nn contains multiple layers like lstm
import torch

class CNN_LSTM(nn.Module):  # inherits from nn module
    def __init__(self): # defne the layers
        super(CNN_LSTM, self).__init__() # allows access to methods of parent class
        # 1D convolution layer: input channels = 4, output channels = 32, kernel size = 3, padding = 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU() #  rectified linear function to add non-linerarity and replaces negative value with zero
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.3)# dropping 30% of neurons to prevent overfitting
        self.fc = nn.Linear(64, 1)  # 64 output from LSTM (hidden size)
        self.sigmoid = nn.Sigmoid() # output  in binary  

    def forward(self, x): # this function shows how the data flows throughout the model 
        # Reshape input to (batch_size, channels, sequence_length) if it's not already
        x = x.unsqueeze(2)  # Add the sequence dimension: (batch_size, features, sequence_length)
        
        # Apply Conv1D layer
        x = self.relu(self.conv1(x))  # (batch_size, 32, sequence_length)
        
        # Now we need to permute to get (batch_size, sequence_length, 32) for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 32)
        
        # Apply LSTM
        x, _ = self.lstm(x)  # (batch_size, sequence_length, 64)
        
        # Apply dropout
        x = self.dropout(x[:, -1, :])  # Take the output of the last time step
        
        # Final fully connected layer and sigmoid for binary classification
        x = self.sigmoid(self.fc(x)).squeeze()  # Output shape (batch_size,)
        
        return x
