import numpy as np
from tqdm import tqdm
import sys

# Usage: python src/utils/get_scale_factor.py <data_path> <vision_model>
# Example: python src/utils/get_scale_factor.py /mnt/sda1/Datasets/chal2525/mmap_data theia_base_cdiv

if len(sys.argv) < 3:
    print("Usage: python get_scale_factor.py <data_path> <vision_model>")
    print("Example: python src/utils/get_scale_factor.py /mnt/sda1/Datasets/chal2525/mmap_data theia_base_cdiv")
    sys.exit(1)

data_path = sys.argv[1]
vision_model = sys.argv[2]

print(f"Computing scale factor for {vision_model} from {data_path}...")

# Memory-map both embedding files
initial_embed = np.load(f"{data_path}/initial_embed_{vision_model}.npy", mmap_mode='r')
target_embed = np.load(f"{data_path}/target_embed_{vision_model}.npy", mmap_mode='r')

print(f"Initial embed shape: {initial_embed.shape}")
print(f"Target embed shape: {target_embed.shape}")

# Online computation using Chan's parallel algorithm
# Process in chunks to avoid loading everything into RAM
chunk_size = 1000
n_samples = initial_embed.shape[0]

total_count = 0
total_mean = 0.0
total_M2 = 0.0

for start in tqdm(range(0, n_samples, chunk_size), desc="Processing chunks"):
    end = min(start + chunk_size, n_samples)
    
    for embed_array in [initial_embed, target_embed]:
        chunk = embed_array[start:end].copy().astype(np.float64)
        
        batch_count = chunk.size
        batch_mean = chunk.mean()
        batch_var = chunk.var()
        batch_M2 = batch_var * batch_count
        
        if total_count == 0:
            total_count = batch_count
            total_mean = batch_mean
            total_M2 = batch_M2
        else:
            delta = batch_mean - total_mean
            new_count = total_count + batch_count
            total_mean = (total_count * total_mean + batch_count * batch_mean) / new_count
            total_M2 = total_M2 + batch_M2 + delta**2 * total_count * batch_count / new_count
            total_count = new_count

print(f"\nTotal values processed: {total_count}")

# Compute final statistics
std = (total_M2 / total_count) ** 0.5
mean = total_mean

print(f"Mean: {mean:.6f}")
print(f"Std: {std:.6f}")
print(f"Suggested scale_factor: {1/std:.6f}")