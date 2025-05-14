import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom dataset class for game data.
class GameDataset(Dataset):
    def __init__(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)
        # Each element in self.data is a tuple: (numeric_state, action)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]
        # Convert the state into a NumPy array then to tensor.
        # Original shape: (15, 10)
        # CNN expects a channel dimension: (1, 15, 10)
        state_array = np.array(state, dtype=np.float32)  # Shape: (15, 10)
        state_tensor = torch.tensor(state_array).unsqueeze(0)  # Shape: (1, 15, 10)
        # Action is typically -1, 0, or 1. Wrap it in a tensor:
        action_tensor = torch.tensor([action], dtype=torch.float32)
        return state_tensor, action_tensor

# Define a simple CNN model.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: (batch, 1, 15, 10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # Output becomes (16, 7, 5) if 15,10 becomes floor((15/2,10/2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # Output becomes (32, 3, 2) (roughly)
        
        # Calculate the flattened dimension: 32 * 3 * 2 = 192.
        self.fc1 = nn.Linear(192, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Output: single steering value.
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor.
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def main():
    # Load the dataset.
    dataset = GameDataset('master_game_dataset.pkl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNN()
    
    # Mean Squared Error Loss (since actions are numeric values)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 100  # Adjust based on your data size and loss convergence
    for epoch in range(epochs):
        total_loss = 0.0
        for state_batch, action_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(state_batch)
            loss = criterion(predictions, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the model.
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("CNN model saved as cnn_model.pth")

if __name__ == "__main__":
    main()
