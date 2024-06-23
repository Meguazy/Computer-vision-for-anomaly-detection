import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

class CustomCNN(nn.Module):
    def __init__(self, in_channels: int, batch_size=32, init_weights=True):
        super(CustomCNN, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(7,7), stride=(2,2), padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        self.dense1 = nn.Linear(20736, 224)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features = 224, out_features = 1)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = x.view(-1, 20736)
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)

        return torch.sigmoid(x)

    def validate(self, loader, model, criterion):  
        torch.set_printoptions(sci_mode=False)
        correct = 0
        total = 0
        running_loss = 0.0        
        counter = 0
        running_f1_score = 0
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(loader):
                inputs, labels = data
                labels = labels.view(labels.shape[0], -1).float()

                outputs = model(inputs)
                outputs = (outputs>0.5).float()

                loss = criterion(outputs, labels)
                total += labels.size(0)

                correct += (labels == outputs).float().sum()
                running_loss = running_loss + loss.item()
                running_f1_score += f1_score(labels, outputs)

                counter += 1

        mean_val_accuracy = float(100 * correct / total)
        mean_val_loss = (running_loss / counter)
        mean_f1_score = running_f1_score / counter

        print(f'Validation Accuracy: {mean_val_accuracy:.4f}')
        print(f'Validation Loss: {mean_val_loss:.4f}')
        print(f'F1 Score: {(mean_f1_score * 100):.4f}')

        return mean_val_accuracy, mean_val_loss, mean_f1_score

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False