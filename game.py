import os
import time
import random
import keyboard  # Install with: pip install keyboard
import numpy as np

# Game configuration
GRID_HEIGHT = 15
GRID_WIDTH = 10
OBSTACLE_CHAR = '0'
CAR_CHAR = 'x'
BOUNDARY_CHAR = '1'
EMPTY_CHAR = ' '


'''Create a helper function that converts your grid (which is a list of lists of characters) into a numerical matrix. For instance:

EMPTY_CHAR (' ') → 0

BOUNDARY_CHAR ('1') → 1

OBSTACLE_CHAR ('0') → 2

CAR_CHAR ('x') → 3'''


def grid_to_numeric(grid):
    mapping = {
        ' ': 0,
        '1': 1,
        '0': 2,
        'x': 3
    }
    numeric_grid = np.array([[mapping.get(cell, 0) for cell in row] for row in grid])
    return numeric_grid

# Initialize the grid with side boundaries only
def init_grid():
    grid = []
    for _ in range(GRID_HEIGHT):
        row = [EMPTY_CHAR] * GRID_WIDTH
        row[0] = BOUNDARY_CHAR      # Left boundary
        row[-1] = BOUNDARY_CHAR     # Right boundary
        grid.append(row)
    return grid

# Draw the grid and display the score
def draw_grid(grid, score):
    os.system('cls' if os.name == 'nt' else 'clear')
    for row in grid:
        print("".join(row))
    print(f"\nScore: {score}")
    print("Steer with 'A' and 'D'. Press 'Q' to quit.")

# Add a new obstacle at a random column in the top row
def add_obstacle(grid):
    col = random.randint(1, GRID_WIDTH - 2)  # Avoid boundaries
    grid[0][col] = OBSTACLE_CHAR

# Update obstacles by moving them one row down.
# If an obstacle would move into the base row and the car isn't there, remove it.
def update_obstacles(grid, car_row):
    # Process rows from second-to-last upward (avoid iterating the base row)
    for row in range(GRID_HEIGHT - 2, -1, -1):
        for col in range(1, GRID_WIDTH - 1):
            if grid[row][col] == OBSTACLE_CHAR:
                # Check if the next row is the car/base row.
                if row + 1 == car_row:
                    # If the cell is empty, clear the obstacle so it doesn't stick.
                    if grid[row + 1][col] == EMPTY_CHAR:
                        grid[row][col] = EMPTY_CHAR
                    else:
                        # If the car is there, allow the obstacle to move in for collision detection.
                        grid[row + 1][col] = OBSTACLE_CHAR
                        grid[row][col] = EMPTY_CHAR
                else:
                    # Move obstacle down if the cell below is empty.
                    if grid[row + 1][col] == EMPTY_CHAR:
                        grid[row + 1][col] = OBSTACLE_CHAR
                    grid[row][col] = EMPTY_CHAR

def main():
    grid = init_grid()
    car_row = GRID_HEIGHT - 1  # Place car at the bottom row
    car_col = GRID_WIDTH // 2
    grid[car_row][car_col] = CAR_CHAR

    score = 0
    frame_count = 0

    # Initialize dataset to collect (grid_state, action) pairs
    dataset = []

    while True:
        if frame_count % 5 == 0:
            add_obstacle(grid)

        draw_grid(grid, score)

        # Decide action based on keyboard input
        action = 0  # 0 means no movement
        new_col = car_col
        if keyboard.is_pressed('a'):
            new_col = car_col - 1
            action = -1
        elif keyboard.is_pressed('d'):
            new_col = car_col + 1
            action = 1
        elif keyboard.is_pressed('q'):
            print("Quitting game.")
            break

        # Check that the new position doesn't hit a side boundary
        if grid[car_row][new_col] != BOUNDARY_CHAR:
            grid[car_row][car_col] = EMPTY_CHAR
            car_col = new_col
            grid[car_row][car_col] = CAR_CHAR

        # Record the state and action for training data:
        # Convert the current grid to a numeric matrix.
        numeric_state = grid_to_numeric(grid)
        dataset.append((numeric_state, action))

        # Move obstacles downward; pass car_row so update_obstacles knows the base row.
        update_obstacles(grid, car_row)

        # Collision detection: if an obstacle occupies the car cell, end the game.
        if grid[car_row][car_col] == OBSTACLE_CHAR:
            draw_grid(grid, score)
            print("\nGame Over! Collision detected.")
            break

        score += 1
        frame_count += 1
        time.sleep(0.1)

# Save dataset to file at the end of the game
    # Save dataset to file at the end of the game
    import pickle, os

    # 1) Ensure data/ exists
    os.makedirs('data', exist_ok=True)

    # 2) Write into data/
    file_path = os.path.join('data', 'game_dataset.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {file_path}")

if __name__ == "__main__":
    main()

