import torch
from torch.utils.data import DataLoader
from models.gps_retinanet import GPSRetinaNet
from utils.dataloader import get_dataloader
from utils.train_utils import train_model
from config import device, num_classes, num_epochs, learning_rate

if __name__ == '__main__':
    # Initialize dataset and dataloader
    root_dir = 'C:\\Users\\devfr\\Downloads\\sign_hunter\\dataset_2'
    dataloader = get_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4)

    # Initialize model
    model = GPSRetinaNet(num_classes)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print(f"Training the model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_model(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
