import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Zeroes are safe paths and ones are obstacles that the agent must avoid.
maze = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 1, 1, 1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 0, 1, 0, 1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
                 [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]])

start = (0, 0)
goal = (9, 9)

# The number of times the agent will attempt to navigate the maze.
num_episodes = 5000

# The learning rate that controls how much new information overrides old
# information.
alpha = 0.1

# The discount factor that gives more weight to immediate rewards.
gamma = 0.9

# Probability of exploration vs. exploitation.
epsilon = 0.5

# Penalize hitting obstacles and slightly penalize each step (to find shortest
# paths). Reward reaching the goal.
reward_fire = -10
reward_goal = 50
reward_step = -1

# Possible actions: left, right, up, and down.
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Q-table initialized to 0. This stores the expected rewards for each state-
# action pair.
Q = np.zeros(maze.shape + (len(actions),))


# Ensure that the agent avoids obstacles and only moves inside the maze.
def is_valid(pos):
    r, c = pos
    if r < 0 or r >= maze.shape[0]:
        return False
    if c < 0 or c >= maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True


# Implement exploration vs. exploitation. Exploration is just taking a random
# action and exploitation is taking the best learned action.
def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    else:
        return np.argmax(Q[state])


rewards_all_episodes = []

# Train the agent using the Q-learning algorithm.
for episode in range(num_episodes):
    state = start
    total_rewards = 0
    done = False

    while not done:
        action_index = choose_action(state)
        action = actions[action_index]

        next_state = (state[0] + action[0], state[1] + action[1])

        if not is_valid(next_state):
            reward = reward_fire
            done = True
        elif next_state == goal:
            reward = reward_goal
            done = True
        else:
            reward = reward_step

        # TODO
        # I do not understand what we are doing here.
        old_value = Q[state][action_index]
        next_max = np.max(Q[next_state]) if is_valid(next_state) else 0
        Q[state][action_index] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_rewards += reward

    # global epsilon
    epsilon = max(0.01, epsilon * 0.995)
    rewards_all_episodes.append(total_rewards)


def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break
        visited.add(state)

        best_action = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])

            # Isn't this conditional just the is_valid function?
            if (0 <= next_state[0] < maze.shape[0] and 0 <= next_state[1] < maze.shape[1] and maze[
                next_state] == 0 and next_state not in visited):
                if Q[state][idx] > best_value:
                    best_value = Q[state][idx]
                    best_action = idx
        if best_action is None:
            break

        move = actions[best_action]

        # We repeat the logic for computing the state + action a lot.
        state = (state[0] + move[0], state[1] + move[1])
        path.append(state)
    return path


optimal_path = get_optimal_path(Q, start, goal, actions, maze)


def plot_maze_with_path(path):
    # green-200 and green-300
    cmap = ListedColormap(['#bbf7d0', '#86efac'])

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    # green-500
    plt.scatter(start[1], start[0], marker='o', color='#22c55e', s=200, label='Start', zorder=5)
    plt.scatter(goal[1], goal[0], marker='*', color='#22c55e', s=300, label='Goal', zorder=5)
    rows, cols = zip(*path)

    # green-700
    plt.plot(cols, rows, color='#15803d', linewidth=4, label='Learned path', zorder=4)

    plt.title('Reinforcement learning maze navigation')
    plt.gca().invert_yaxis()
    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_maze_with_path(optimal_path)


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Total rewards per episode')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.grid(True)
    plt.show()


plot_rewards(rewards_all_episodes)
