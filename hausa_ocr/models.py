"""IX-World: architectures retained from the historical V9--V12 scripts."""
import torch.nn as nn
CNN_DROPOUT = 0.05

class CRNN6(nn.Module):

    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, None)))
        self.rnn = nn.LSTM(input_size=384, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.classifier(x)

class CRNN12(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)), nn.Dropout2d(CNN_DROPOUT), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)), nn.AdaptiveAvgPool2d((1, None)))
        self.rnn = nn.LSTM(input_size=384, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.classifier(x)

def build_model(version, classes):
    if version not in ("v9", "v10", "v11", "v12"):
        raise ValueError(f"Version non implementee : {version}")
    if version == "v12":
        return CRNN12(classes)
    return CRNN6(classes, input_channels=2 if version == "v11" else 1)
