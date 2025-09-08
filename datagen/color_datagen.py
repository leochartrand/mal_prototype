import numpy as np
import pickle

moves = ["up", "down", "left", "right"]
colors = ["red", "green", "blue"]
color_to_channel = {"red":0, "green":1, "blue":2}

def move_2x2_dot(pos, move):
    """Move a 2x2 dot in the specified direction"""
    if move == "up":
        return (pos[0] - 1, pos[1])
    elif move == "down":
        return (pos[0] + 1, pos[1])
    elif move == "left":
        return (pos[0], pos[1] - 1)
    elif move == "right":
        return (pos[0], pos[1] + 1)
    return pos

def block_crosses_border(pos):
    """Check if a 2x2 block starting at pos would cross grid boundaries"""
    r, c = pos
    return r < 0 or c < 0 or r + 1 >= 8 or c + 1 >= 8

def blocks_overlap(pos1, pos2):
    """Check if two 2x2 blocks overlap"""
    r1, c1 = pos1
    r2, c2 = pos2
    return not (r2 >= r1+2 or r1 >= r2+2 or c2 >= c1+2 or c1 >= c2+2)

def place_2x2_dot(grid, color, pos):
    """Place a 2x2 dot with top-left at pos, only if no overlap exists"""
    r, c = pos
    
    # First check if the 2x2 block would fit in the grid
    if r + 1 >= 8 or c + 1 >= 8:
        return False  # Would go out of bounds
    
    # Check if any pixel in the 2x2 area is already occupied
    for dr in range(2): 
        for dc in range(2):
            if grid[r+dr, c+dc, :].any():  # Check all channels for existing pixels
                return False  # Overlap detected, don't place
    
    # No overlap, place the 2x2 dot
    for dr in range(2):
        for dc in range(2):
            grid[r+dr, c+dc, color_to_channel[color]] = 1

    return True  # Successfully placed

def generate_color_data(n_samples=1000):
    data = []
    for z in range(n_samples):
        # Random starting position
        pos = (np.random.randint(0, 7), np.random.randint(0, 7))  
        pos2 = (np.random.randint(0, 7), np.random.randint(0, 7))
        while blocks_overlap(pos, pos2):
            pos2 = (np.random.randint(0, 7), np.random.randint(0, 7))

        color = colors[np.random.randint(0, 3)] # Random color channel for first dot
        color2 = colors[np.random.randint(0, 3)] # Random color channel for second dot
        while color2 == color:
            color2 = colors[np.random.randint(0, 3)]

        # RGB channels for two dots
        before = np.zeros((8, 8, 3)) # [H, W, C]

        # Place before dots
        place_2x2_dot(before, color, pos)
        place_2x2_dot(before, color2, pos2)
        
        # Try different moves until we find a valid one
        valid_moves = []
        for move in moves:
            new_pos2 = move_2x2_dot(pos2, move)
            if (new_pos2 != pos2 and 
                not blocks_overlap(pos, new_pos2) and 
                not block_crosses_border(new_pos2)):  # Valid move with no overlap
                valid_moves.append((move, new_pos2))
        
        if not valid_moves:
            continue  # Skip this sample if no valid moves exist
            
        move, new_pos2 = valid_moves[np.random.randint(len(valid_moves))]
        
        after = np.zeros((8, 8, 3)) # [C, H, W]
        place_2x2_dot(after, color, pos)
        place_2x2_dot(after, color2, new_pos2)

        data.append((before, after, move + " " + color2))
    return data

data = generate_color_data(10000)

# #visualize a sample grid of after images
import matplotlib.pyplot as plt
grid_size = 7
for i in range(grid_size):
    for j in range(grid_size):
        plt.subplot(grid_size, grid_size, i*grid_size + j + 1)
        plt.axis('off') # Turn off axiss for clarity      
        sample = data[i*grid_size + j][1] # put after image in subplot
        plt.imshow(sample)
plt.show()

# visualize a list of 5 samples with their commands
# side by side with before and after
# in new figure
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.title("Before")
    sample = data[i]
    plt.imshow(sample[0])
    plt.axis('off')
    plt.subplot(3, 5, i + 6)
    plt.title(f"Cmd: {sample[2]}")
    plt.axis('off')
    plt.subplot(3, 5, i + 11)
    plt.title("After")
    plt.imshow(sample[1])
    plt.axis('off')
plt.show()


# write to a npy file
with open("data/color_data.pkl", "wb") as f:
    pickle.dump(data, f)
