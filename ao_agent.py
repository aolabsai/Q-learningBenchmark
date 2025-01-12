import ao_core as ao
from arch__ao_agent import Arch
import random
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Grid environment setup
grid_size = 5
start = (0, 0)
goal = (4, 4)
num_obs = 3

# Generate obstacles
obs = set()
while len(obs) < num_obs:
    obstacle = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    if obstacle != start and obstacle != goal:
        obs.add(obstacle)

print("Obstacles:", obs)

# Action mapping
action_mapping = {
    (0, 0): (-1, 0),   # Move up
    (1, 0): (1, 0),    # Move down
    (0, 1): (0, -1),   # Move left
    (1, 1): (0, 1)     # Move right
}

# Encode position in binary
def encode_position_binary(x, y):
    x_bin = format(x, '03b')
    y_bin = format(y, '03b')
    return [int(bit) for bit in x_bin + y_bin]

# Check if a position is valid
def is_valid(pos):
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size and pos not in obs

# Visualization function
def visualize_grid(state):
    grid = np.ones((grid_size, grid_size))  # 1 = free space
    for obstacle in obs:
        grid[obstacle] = 0  # 0 = obstacle

    # Mark start, goal, and agent positions
    grid[start] = 0.5  # Start (green)
    grid[goal] = 0.2   # Goal (red)
    grid[state] = 0.8  # Agent (blue)

    plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.title("Agent's Pathfinding")
    plt.pause(0.5)
    plt.clf()

# Initialize agent
agent = ao.Agent(Arch)

# Solve the grid
timed_out = False
solved = False
state = start
steps = 0

plt.ion()
while not solved and not timed_out:
    steps += 1

    # Show grid visualization
    visualize_grid(state)

    input_to_agent = encode_position_binary(*state)
    response = agent.next_state(input_to_agent).tolist()
    response_tuple = tuple(response)

    if response_tuple in action_mapping:
        dx, dy = action_mapping[response_tuple]
        new_state = (state[0] + dx, state[1] + dy)

        if not is_valid(new_state):
            agent.next_state(input_to_agent, Cneg=True)  # Pain signal for invalid move
            print("Pain: Hit an obstacle. Returning to start.")
            state = start  # Reset to starting position
            
        elif new_state == goal:
            agent.next_state(input_to_agent, Cpos=True)  # Reward for reaching the goal
            solved = True
            print("Goal reached in", steps, "steps!")
        else:
            state = new_state
    else:
        print("Invalid response from the agent.")
        agent.next_state(input_to_agent, Cneg=True)

    # Stop condition to prevent infinite loops
    if steps > 100:
        timed_out = True
        print("Agent timed out.")

plt.ioff()
plt.show()
