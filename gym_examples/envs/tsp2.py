import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import itertools
#from gym.envs.registration import register

class TSPEnvironment(gym.Env):
    def __init__(self, num_cities):
        super(TSPEnvironment, self).__init__()
        self.num_cities = num_cities
        self.city_coordinates = np.random.rand(num_cities, 2)   # Random 2D coordinates for cities
        self.city_coordinates = self.city_coordinates *100
        self.distance_matrix = self.calculate_distance_matrix()

        self.current_city = 0
        self.visited_cities = set([0])  # Set to keep track of visited cities
        #self.optimal_tour = self.get_optimal_tour()
        # Gym spaces
        self.action_space = spaces.Discrete(num_cities)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.city_ids  = list(range(0, num_cities ))
        self.total_length = 0

    def reset(self):
        self.current_city = 0
        self.visited_cities = set([0])
        #return self.city_coordinates[self.current_city]
        return self.current_city

    def calculate_distance_matrix(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distances[i, j] = np.linalg.norm(self.city_coordinates[i] - self.city_coordinates[j])
        return distances

    def step(self, action,i,t):
        if action not in self.visited_cities:
            reward = 1/self.distance_matrix[self.current_city, action] + t * self.distance_matrix[self.current_city, action]
            self.visited_cities.add(action)
            self.current_city = action
            self.total_length = self.total_length + self.distance_matrix[self.current_city, action]
        else:
            reward = -self.distance_matrix[self.current_city, action]  # Penalize revisiting a city

        if i<75000:
          done = len(self.visited_cities) == self.num_cities
          if done:
              self.total_length = 0
              missing_ids = [city_id for city_id in self.city_ids if city_id not in self.visited_cities]
              if not missing_ids:
                  reward= 10*self.num_cities
                  done = True
              else:
                reward = -10 * len(missing_ids)
                done = False
        else:
          if self.total_length ==15:
             missing_ids = [city_id for city_id in self.city_ids if city_id not in self.visited_cities]
             if not missing_ids:
              reward= 10*self.num_cities
              done =True
             else:
              reward = -10 * len(missing_ids)
              done = False
       # next_state = self.city_coordinates[self.current_city]
     #   return self.city_coordinates[self.current_city], reward,done {}
        return action, reward,done,{}
    def get_optimal_tour(self):
        points = list(range(self.num_cities))
        all_distances = self.distance_matrix
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j]) for k in S if
                                     k != 0 and k != j])
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        tour = np.array(res[1])
        length =sum(self.distance_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
        return length,tour

    def get_optimal_reward(self):
        optimal_reward = sum(self.distance_matrix[self.optimal_tour[i], self.optimal_tour[i+1]] for i in range(len(self.optimal_tour)-1))
        optimal_reward += self.distance_matrix[self.optimal_tour[-1], self.optimal_tour[0]]
        return optimal_reward
