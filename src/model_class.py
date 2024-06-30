import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# Custom CNN class for the model
class CustomCNN(nn.Module):
    # Initialize the model
    def __init__(self, in_channels: int, batch_size=32, init_weights=True):
        super(CustomCNN, self).__init__()
        # Set the batch size
        self.batch_size = batch_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        
        # Maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # Fully connected layers
        self.dense1 = nn.Linear(20736, 224)
        # Dropout layer with 20% probability
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features = 224, out_features = 1)

        # Initialize the weights
        if init_weights:
            self._initialize_weights()

    # Function to initialize the weights
    def _initialize_weights(self):
        # Loop over the modules in the model
        for m in self.modules():
            # If the module is a convolutional layer initialize the weights using kaiming distribution
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # If the module is a batch normalization layer initialize the weights using normal distribution
            # with mean 0 and standard deviation 0.01
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # Forward pass
    def forward(self, x):
        # Convolutional layers with ReLU activation function and maxpooling
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))

        # Flatten the output
        x = x.view(-1, 20736)

        # Fully connected layers with ReLU activation function
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)

        # Sigmoid activation function for binary classification
        return torch.sigmoid(x)

    # Function to validate the model
    def validate(self, loader, model, criterion):
        torch.set_printoptions(sci_mode=False)
        # Set the model to evaluation mode
        model.eval()
        # Initialize the variables for the correct predictions, total predictions, running loss and counter
        correct = 0
        total = 0
        running_loss = 0.0
        counter = 0
        running_f1_score = 0
        # Loop over the data in the loader and calculate the loss and accuracy
        with torch.no_grad():
            for _, data in enumerate(loader):
                # Get the inputs and labels from the data
                inputs, labels = data
                labels = labels.view(labels.shape[0], -1).float()

                # Get the outputs from the model and calculate the loss
                outputs = model(inputs)
                outputs = (outputs>0.5).float()
                loss = criterion(outputs, labels)

                # Calculate the total number of predictions and the number of correct predictions
                total += labels.size(0)
                correct += (labels == outputs).float().sum()

                # Calculate the running loss and running f1 score
                running_loss = running_loss + loss.item()
                running_f1_score += f1_score(labels, outputs)

                # Increment the counter
                counter += 1

        # Calculate the mean accuracy, loss and f1 score over the dataset
        mean_val_accuracy = float(100 * correct / total)
        mean_val_loss = (running_loss / counter)
        mean_f1_score = running_f1_score / counter

        # Return the mean accuracy, loss and f1 score
        return mean_val_accuracy, mean_val_loss, mean_f1_score

# Early stopping class
class EarlyStopper:
    # Initialize the early stopper
    def __init__(self, patience=1, min_delta=0):
        # Set the patience threshold and the minimum delta
        self.patience = patience
        self.min_delta = min_delta
        # Initialize the counter and the minimum validation loss
        self.counter = 0
        self.min_validation_loss = float('inf')

    # Function to check if the model should stop training
    def early_stop(self, validation_loss):
        # If the validation loss is less than the minimum validation loss, reset the counter
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # If the validation loss is greater than the minimum validation loss plus the minimum delta increment the counter
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # If the counter is greater than or equal to the patience return True
            if self.counter >= self.patience:
                return True
        return False