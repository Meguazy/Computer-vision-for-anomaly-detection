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

writer = SummaryWriter()

tr_transf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()])

train_set = torchvision.datasets.ImageFolder(r'archive/casting_data/casting_data/train/', transform=tr_transf)

train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, val_size])

# Hyper-parameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 20

val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = CustomCNN(1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

early_stopper = EarlyStopper(patience=3, min_delta=0)

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader),total=len(train_loader))
    loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, targets) in loop:
        # backward
        optimizer.zero_grad()

        scores = model(data)

        targets = targets.view(targets.shape[0], 1)
        loss = criterion(scores, targets.float())
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
        #Update Progress bar
        loop.set_description(f'Epoch [{epoch+1}]')
        loop.set_postfix(loss = loss.item())

        # Update Tensorboard
        writer.add_scalar("loss x epoch", loss/len(train_loader), epoch)
        grid = torchvision.utils.make_grid(data)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, data)

    val_acc, val_loss = model.validate(val_loader, model, criterion)
    writer.add_scalar("val loss x epoch", val_loss, epoch)
    writer.add_scalar("val accuracy x epoch", val_acc, epoch)

    # Save the model
    print(f'Saving model for epoch {epoch + 1}...')
    PATH = f'./model/cnn_{epoch + 1}.pth'
    torch.save(model.state_dict(), PATH)
    if early_stopper.early_stop(val_loss):
        break

writer.close()