import numpy as np
import random
import matplotlib.pyplot as plt

# Grid environment setup
grid_size = 5
start = (0, 0)
goal = (4, 4)
obs = [(1, 1), (1, 3), (3, 1)]

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
exploration_decay = 0.99
epochs = 1000

# Initialize Q-table
Q_table = np.zeros((grid_size, grid_size, 4))

#map action
actions = ['up', 'down', 'left', 'right']
action_mapping = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

#give reward
def give_reward(position):
    if position == goal:
        return 10
    elif position in obs:
        return -10
    else:
        return -1

#check if the positon chosen is valid
def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size

#choose the action 
def choose(state):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(actions)
    else:
        x, y = state
        return actions[np.argmax(Q_table[x, y])]

# Q-learning process
for epoch in range(epochs):
    state = start
    while state != goal:
        x, y = state
        action = choose(state)
        dx, dy = action_mapping[action]
        new_state = (x + dx, y + dy)

        if not is_valid(new_state) or new_state in obs:
            new_state = state
            reward = -10
        else:
            reward = give_reward(new_state)

        new_x, new_y = new_state

        Q_table[x, y, actions.index(action)] += learning_rate * (
            reward + discount_factor * np.max(Q_table[new_x, new_y]) - Q_table[x, y, actions.index(action)]
        )

        state = new_state



def trace_path():
    path = []
    state = start
    while state != goal:
        path.append(state)
        x, y = state
        action = actions[np.argmax(Q_table[x, y])]
        dx, dy = action_mapping[action]
        state = (x + dx, y + dy)
    path.append(goal)
    return path


def visualize_grid(path):
    grid = np.zeros((grid_size, grid_size))

    # Mark start, goal, and obstacles
    grid[start] = 0.5   #start
    grid[goal] = 0.8    #goal
    for obstacle in obs:
        grid[obstacle] = 1  #obstacle

    # Create plot
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Plot the path
    for (x, y) in path:
        ax.text(y, x, '●', ha='center', va='center', color='red')


    plt.title("Path taken by agent")
    plt.show()

# Get and visualize the learned path
path = trace_path()
print("\nLearned Path:")
print(path)
visualize_grid(path)
