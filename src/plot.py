import pandas as pd
import matplotlib.pyplot as plt

# Load the data for the training loss
df1 = pd.read_csv('./tensorboard_outputs/loss_x_epoch.csv', usecols=['Step', 'Value'])
df1.columns = ['Step', 'train_loss']
df1 = df1.groupby('Step')['train_loss'].min().reset_index()

# Load the data for the validation accuracy
df2 = pd.read_csv('./tensorboard_outputs/val_accuracy_x_epoch.csv', usecols=['Step', 'Value'])
df2.columns = ['Step', 'val_accuracy']

# Load the data for the validation loss
df3 = pd.read_csv('./tensorboard_outputs/val_loss_x_epoch.csv', usecols=['Step', 'Value'])
df3.columns = ['Step', 'val_loss']

# Merge the dataframes
df = df1.merge(df2,how='left', left_on='Step', right_on='Step').merge(df3,how='left', left_on='Step', right_on='Step')

# Plot the data
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(df['Step'], df['train_loss'] * 10000, label='train_loss')
plt.plot(df['Step'], df['val_loss'], label='val_loss')
plt.plot(df['Step'], df['val_accuracy'], label='val_accuracy')
plt.xticks(df['Step'])
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Train x Epoch')
plt.legend()
plt.show()