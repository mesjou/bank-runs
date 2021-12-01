import random
from abc import ABC

import numpy as np


class QLearner(ABC):
    def __init__(self, observation_space: int, action_space: int):
        assert isinstance(action_space, int)
        self.action_space = action_space
        self.q_table = np.zeros([observation_space, action_space])

        # Hyperparameters
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 0.2

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_space - 1)  # Explore action space
        else:
            action = np.argmax(self.q_table[state])  # Exploit learned values
        return action

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def set_learning_rate(self, alpha):
        self.alpha = alpha

    def get_learning_rate(self):
        return self.alpha
