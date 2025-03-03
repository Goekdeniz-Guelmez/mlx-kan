# Copyright © 2024-2025 Gökdeniz Gülmez

# Import the KAN model class
from kan import KAN

# Define the folder path where the model is stored
folder_path = "/path/to/model/folder"  # Update this path to the actual location of your model

# Load the pre-trained KAN model from the specified folder
model = KAN.load_model(folder_path)

# Print the details of the loaded model
print(model)