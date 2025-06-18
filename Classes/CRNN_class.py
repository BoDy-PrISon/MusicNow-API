import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# === Базовый класс CRNN для мультикласса ===
class CRNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10, dropout_rate=0.3):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout_rate)

        self.gru = nn.GRU(
            input_size=128 * (n_mels // 8),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        self.fc1 = nn.Linear(128 * 2, 64)
        self.drop4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])
        x = self.drop4(x)
        x = self.fc2(x)
        return x


# === CRNN для IRMAS (multi-label классификация инструментов) ===
class CRNN_IRMAS(nn.Module):
    def __init__(self, n_mels=128, n_classes=11):  # 11 инструментов IRMAS
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        self.gru = nn.GRU(
            input_size=128 * (n_mels // 8),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.fc1 = nn.Linear(128 * 2, 64)
        self.drop4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, n_classes)
        # Для multi-label используем sigmoid, а не softmax

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])
        x = self.drop4(x)
        x = self.fc2(x)

        # Для multi-label classification используем sigmoid
        return torch.sigmoid(x)


class AttentionCRNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10):
        super(AttentionCRNN, self).__init__()
        
        # Базовая CRNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        self.classifier = nn.Linear(256, n_classes)
    
    def forward(self, x):
        # CNN
        x = self.conv_layers(x)
        
        # Reshape для LSTM
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, width, channels * height)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResCRNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10):
        super(ResCRNN, self).__init__()
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, n_classes)