import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

# Grid configuration constants (same as your game.py)
GRID_HEIGHT = 15
GRID_WIDTH = 10
OBSTACLE_CHAR = '0'
CAR_CHAR = 'x'
BOUNDARY_CHAR = '1'
EMPTY_CHAR = ' '

# Convert the grid to a numerical matrix
def grid_to_numeric(grid):
    mapping = {
        ' ': 0,
        '1': 1,
        '0': 2,
        'x': 3
    }
    numeric_grid = np.array([[mapping.get(cell, 0) for cell in row] for row in grid])
    return numeric_grid

# Initialize the grid with boundaries
def init_grid():
    grid = []
    for _ in range(GRID_HEIGHT):
        row = [EMPTY_CHAR] * GRID_WIDTH
        row[0] = BOUNDARY_CHAR
        row[-1] = BOUNDARY_CHAR
        grid.append(row)
    return grid

# Display the grid and score
def draw_grid(grid, score):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print("".join(row))
    print(f"\nScore: {score}")

# Add an obstacle randomly in the top row (avoiding boundaries)
def add_obstacle(grid):
    col = random.randint(1, GRID_WIDTH - 2)
    grid[0][col] = OBSTACLE_CHAR

# Update obstacles: move them down the grid, and manage collisions.
def update_obstacles(grid, car_row):
    for row in range(GRID_HEIGHT - 2, -1, -1):
        for col in range(1, GRID_WIDTH - 1):
            if grid[row][col] == OBSTACLE_CHAR:
                if row + 1 == car_row:
                    if grid[row + 1][col] == EMPTY_CHAR:
                        grid[row][col] = EMPTY_CHAR
                    else:
                        grid[row + 1][col] = OBSTACLE_CHAR
                        grid[row][col] = EMPTY_CHAR
                else:
                    if grid[row + 1][col] == EMPTY_CHAR:
                        grid[row + 1][col] = OBSTACLE_CHAR
                    grid[row][col] = EMPTY_CHAR

# Define the same CNN model architecture as used during training.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # After pooling, the feature map should roughly be of size (32, 3, 2)
        self.fc1 = nn.Linear(32 * 3 * 2, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
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
    # Load the trained CNN model.
    model = CNN()
    model_file = os.path.join('models', 'cnn_model.pth')
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set model to evaluation mode.

    # Initialize the grid and place the car.
    grid = init_grid()
    car_row = GRID_HEIGHT - 1
    car_col = GRID_WIDTH // 2
    grid[car_row][car_col] = CAR_CHAR

    score = 0
    frame_count = 0

    while True:
        if frame_count % 5 == 0:
            add_obstacle(grid)

        draw_grid(grid, score)

        # Convert the current grid state for inference:
        numeric_state = grid_to_numeric(grid)
        # Convert the state to a tensor with shape (1, 1, 15, 10)
        state_tensor = torch.tensor(numeric_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            predicted_value = model(state_tensor).item()

        # Decision thresholding:
        if predicted_value < -0.3:
            action = -1  # steer left
        elif predicted_value > 0.3:
            action = 1   # steer right
        else:
            action = 0   # no movement

        new_col = car_col + action
        if grid[car_row][new_col] != BOUNDARY_CHAR:
            grid[car_row][car_col] = EMPTY_CHAR
            car_col = new_col
            grid[car_row][car_col] = CAR_CHAR

        update_obstacles(grid, car_row)

        # Collision detection: if the car cell gets an obstacle, game over.
        if grid[car_row][car_col] == OBSTACLE_CHAR:
            draw_grid(grid, score)
            print("\nGame Over! Collision detected.")
            break

        score += 1
        frame_count += 1
        time.sleep(0.1)

if __name__ == "__main__":
    main()

