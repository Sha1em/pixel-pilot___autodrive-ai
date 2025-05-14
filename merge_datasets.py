import pickle
import os

def load_dataset(file_path):
    """
    Load dataset from a pickle file.
    Returns an empty list if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        return []

def save_dataset(dataset, file_path):
    """
    Save dataset (a Python object) to a pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {file_path} with {len(dataset)} records.")

def merge_datasets(master_file, new_file):
    """
    Merge the master dataset with new data.
    """
    # Load the master dataset (if it exists; otherwise, an empty list)
    master_data = load_dataset(master_file)
    print(f"Master dataset loaded. It has {len(master_data)} records.")

    # Load the new session dataset
    new_data = load_dataset(new_file)
    print(f"New session data loaded. It has {len(new_data)} records.")

    # Concatenate the lists
    merged_data = master_data + new_data
    print(f"After merging, total records: {len(merged_data)}.")

    # Save the merged dataset back to the master file
    save_dataset(merged_data, master_file)

if __name__ == "__main__":
    # Define your file names.
    # Use a permanent name for your master dataset.
    # point at the files inside data/
    os.makedirs('data', exist_ok=True)
    master_file = os.path.join('data', 'master_game_dataset.pkl')
    new_file   = os.path.join('data', 'game_dataset.pkl')
    
    merge_datasets(master_file, new_file)
