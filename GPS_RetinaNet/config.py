import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 171  # Total number of classes in ARTSv2 (When i checked)
num_epochs = 10
learning_rate = 1e-4
