import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerFCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerFCModel, self).__init__()

        # First fully connected layer with batch normalization and ReLU
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()

        # 2nd fully connected layer with batch normalization and ReLU
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.batch_norm3 = nn.BatchNorm1d(output_size)
        self.relu3 = nn.ReLU()

        # Softmax layer for binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = torch.reshape(x, (x.size()[0], -1))
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.softmax(x)

        return x
    

# CNN + STFT - Inspired from sensors paper provided
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 24, kernel_size=12, padding='same')
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(24, 48, kernel_size=8, padding='same')
        self.bn2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(48, 96, kernel_size=4, padding='same')
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(96 * 4 * 4, 2) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        # print(x.size())

        # Fully Connected Layer
        x = self.fc(x)
        x = self.softmax(x)
        
        return x

# LSTM + timeseries - Inspired from sensors paper provided
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.05)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Sigmoid activation function for binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = out[:, -1, :].squeeze()

        out = self.dropout(out)
        out = self.fc(out)

        out = self.softmax(out)

        return out