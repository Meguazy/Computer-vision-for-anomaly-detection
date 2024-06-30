import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim                              # Optimizers like Adam
from tqdm import tqdm                                    # To print the Progress bar

from model_class import CustomCNN
from model_class import EarlyStopper

# Tensorboard initialization
writer = SummaryWriter()

# Transformations for the images
tr_transf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()])

# Load the dataset
train_set = torchvision.datasets.ImageFolder(r'archive/casting_data/casting_data/train/', transform=tr_transf)

# Split the dataset into train and validation
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, val_size])

# Load the data into the DataLoader
batch_size = 32
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Hyper-parameters
learning_rate = 1e-4
num_epochs = 20

# Initialize the model
model = CustomCNN(1)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping initialization
early_stopper = EarlyStopper(patience=3, min_delta=0)

# Training loop
model.train()
for epoch in range(num_epochs):
    # Progress bar for the training loop
    loop = tqdm(enumerate(train_loader),total=len(train_loader))
    loss = 0
    correct = 0

    # Loop over the dataset
    for batch_idx, (data, targets) in loop:
        # Reseting the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        scores = model(data)
        
        # Calculate the loss
        targets = targets.view(targets.shape[0], 1)
        loss = criterion(scores, targets.float())

        # Backward pass
        loss.backward()

        # Adam optimizer step
        optimizer.step()
        
        #Update Progress bar
        loop.set_description(f'Epoch [{epoch+1}]')
        loop.set_postfix(loss = loss.item())

        # Update Tensorboard with the loss
        writer.add_scalar("loss x epoch", loss/len(train_loader), epoch)
        # Adding the images to tensorboard
        grid = torchvision.utils.make_grid(data)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, data)

    # Validation calculation over the epoch
    val_acc, val_loss, val_f1 = model.validate(val_loader, model, criterion)
    # Update Tensorboard with the validation metrics
    writer.add_scalar("val loss x epoch", val_loss, epoch)
    writer.add_scalar("val accuracy x epoch", val_acc, epoch)
    writer.add_scalar("val f1 x epoch", val_f1, epoch)

    # Early stopping if the validation loss is not improving
    if early_stopper.early_stop(val_loss):
        print(f'Early stopping at epoch {epoch+1}')
        break

# Close the tensorboard writer
writer.close()

# Save the model
print(f'Saving model...')
PATH = f'./model/cnn.pth'
torch.save(model.state_dict(), PATH)