import os
import shutil
import re

# Define the base directory and the target output directories
base_dir = r"/home/johannes/LION/LION/data_loaders/2deteCT/processed"
output_sinograms = os.path.join(base_dir, "all_sinograms")
output_reconstructions = os.path.join(base_dir, "all_reconstructions")

# Create the new folders if they don't exist yet
os.makedirs(output_sinograms, exist_ok=True)
os.makedirs(output_reconstructions, exist_ok=True)

print("Starting to sort and rename files...")

# Iterate through everything inside the processed folder
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    
    # Skip our new output directories and ensure we are looking at slice folders
    if item in ["all_sinograms", "all_reconstructions"] or not os.path.isdir(item_path):
        continue
    
    # Extract the slice number from the folder name (e.g., "slice_123" -> "123")
    # If your folders are just numbers, it will extract the whole number.
    slice_match = re.search(r'\d+', item)
    if slice_match:
        slice_num = slice_match.group()
    else:
        # Fallback to the whole folder name if no number is found
        slice_num = item 

    # --- Process Sinogram (Mode 1) ---
    sinogram_source = os.path.join(item_path, "mode1", "sinogram.npy")
    if os.path.exists(sinogram_source):
        new_sino_name = f"sinogram_slice_{slice_num}.npy"
        shutil.copy2(sinogram_source, os.path.join(output_sinograms, new_sino_name))
    else:
        print(f"Warning: sinogram.npy missing in {item}/mode1")

    # --- Process Reconstruction (Mode 3) ---
    recon_source = os.path.join(item_path, "mode3", "reconstruction.npy")
    if os.path.exists(recon_source):
        new_recon_name = f"reconstruction_slice_{slice_num}.npy"
        shutil.copy2(recon_source, os.path.join(output_reconstructions, new_recon_name))
    else:
        print(f"Warning: reconstruction.npy missing in {item}/mode3")

print("\nTask complete! Your folders are ready:")
print(f"-> Sinograms: {output_sinograms}")
print(f"-> Reconstructions: {output_reconstructions}")