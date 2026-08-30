import torch

# Try loading the zip file directly
checkpoint = torch.load("C:/Users/Amjad/Downloads/best_6500.zip", map_location='cpu')

# Resave it explicitly as a .pt file
torch.save(checkpoint, 'C:/Users/Amjad/Downloads/best_6500.pt')