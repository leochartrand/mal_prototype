import numpy as np
import pickle

commands = ["up", "down", "left", "right"]

def generate_dot_data(n_samples=1000):
    data = []
    for _ in range(n_samples):
        # Random starting position
        pos = (np.random.randint(0, 8), np.random.randint(0, 8))
        before = np.zeros((8, 8))
        before[pos] = 1
        
        # Random command
        cmd = np.random.choice(commands)
        # Move the dot according to the command
        if cmd == "up" and pos[0] > 0:
            pos = (pos[0] - 1, pos[1])
        elif cmd == "down" and pos[0] < 7:
            pos = (pos[0] + 1, pos[1])
        elif cmd == "left" and pos[1] > 0:
            pos = (pos[0], pos[1] - 1)
        elif cmd == "right" and pos[1] < 7:
            pos = (pos[0], pos[1] + 1)
        
        after = np.zeros((8, 8))
        after[pos] = 1
        data.append((before, after, cmd))
    return data

data = generate_dot_data(4096)

# #visualize a sample
import matplotlib.pyplot as plt
sample = data[0]
plt.subplot(1, 3, 1)
plt.title("Before")
plt.imshow(sample[0], cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Command: " + sample[2])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title("After")
plt.imshow(sample[1], cmap='gray')
plt.show()

# write to a npy file
with open("data/dot_data.pkl", "wb") as f:
    pickle.dump(data, f)
