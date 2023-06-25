import torch
import numpy as np


class ReplayMemory():

    def __init__(self, memory_size, sample_size):
        self.memory_counter = 0
        self.sample_size = sample_size
        self.memory_size = memory_size
        self.state_memory = torch.FloatTensor(self.memory_size, sample_size)
        self.action_memory = torch.FloatTensor(self.memory_size, 1)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size, sample_size)

    def store(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size

        # state = torch.tensor(np.array([s[3], s[4]]))
        self.state_memory[index] = s

        # action = torch.tensor(np.array([s_[3] - s[3], s_[4] - s[4]]))
        self.action_memory[index] = a[1]

        self.reward_memory[index] = torch.FloatTensor([r])

        # state_ = torch.tensor(np.array([s_[3], s_[4]]))
        self.state__memory[index] = s_

        self.memory_counter += 1

    def sample(self, size):
        sample_index = np.random.choice(self.memory_size, size)
        state_sample = torch.FloatTensor(size, self.sample_size).cuda()
        action_sample = torch.FloatTensor(size, self.sample_size).cuda()
        reward_sample = torch.FloatTensor(size, 1).cuda()
        state__sample = torch.FloatTensor(size, self.sample_size).cuda()

        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]
            action_sample[index] = self.action_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]
            state__sample[index] = self.state__memory[sample_index[index]]

        return state_sample, action_sample, reward_sample, state__sample