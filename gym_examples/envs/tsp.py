import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
#from gym.envs.registration import register

class TSPEnvironment(gym.Env):
    def __init__(self, num_cities):
        super(TSPEnvironment, self).__init__()
        self.num_cities = num_cities
        self.city_coordinates = np.random.rand(num_cities, 2)   # Random 2D coordinates for cities
        self.distance_matrix = self.calculate_distance_matrix()

        self.current_city = 0
        self.visited_cities = set([0])  # Set to keep track of visited cities

        # Gym spaces
        self.action_space = spaces.Discrete(num_cities)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

        # Visualization
        self.fig, self.ax = plt.subplots()

    def reset(self):
        self.current_city = 0
        self.visited_cities = set([0])
        return self.city_coordinates[self.current_city]

    def calculate_distance_matrix(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distances[i, j] = np.linalg.norm(self.city_coordinates[i] - self.city_coordinates[j])
        return distances

    def step(self, action):
        if action not in self.visited_cities:
            reward = self.distance_matrix[self.current_city, action]
            self.visited_cities.add(action)
            self.current_city = action
        else:
            reward = 0  # Penalize revisiting a city

        done = len(self.visited_cities) == self.num_cities
        # next_state = self.city_coordinates[self.current_city]
        next_state = self.current_city

        self.ax.clear()
        self.ax.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], color='blue', label='Cities')
        self.ax.scatter(*next_state, color='green' if not done else 'red', marker='*' if not done else 'x', s=200, label='Next City')
        if not done:
            self.ax.scatter(*self.city_coordinates[[list(self.visited_cities)[0]]].T, color='orange', marker='o', s=200, label='Start City')
        self.ax.legend()
        plt.pause(1)  # To give time for visualization

        return next_state, reward, done, {}

def render(self, mode='human'):
    plt.figure(figsize=(6, 6))
    plt.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], color='blue', label='Cities')
    plt.scatter(self.city_coordinates[self.current_city, 0], self.city_coordinates[self.current_city, 1], color='red', label='Current City')
    plt.legend()
    plt.title('Traveling Salesman Problem')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()



