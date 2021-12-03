#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random

import torch
import torch.nn
import torch.optim
import torch.nn.functional
from torch.autograd import Variable

class Network(torch.nn.Module):
    
    def __init__(self, input_size, nb_action):
        
        super(Network, self).__init__()
        
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = torch.nn.Linear(input_size, 30)
        self.fc2 = torch.nn.Linear(30, nb_action)
        
    def forward(self, state):
        
        x = torch.nn.functional.relu(self.fc1(state))
        
        Q_values = self.fc2(x)
        
        return Q_values
    
class ReplayMemory(object):
    
    def __init__(self, capacity):
        
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        
        self.memory.append(event)
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        
        samples = zip(* random.sample(self.memory, batch_size))
        
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
class DQN(object):
    
    def __init__(self, input_size, nb_action, gamma):
        
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = torch.optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        self.last_action = 0
        self.last_reward = 0
        
    def get_action(self, state):
        
        probabilities = torch.nn.functional.softmax(self.model(Variable(state)) * 100)
        action = probabilities.multinomial(len(probabilities))
        
        return action.data[0, 0]
    
    def learn(self, states, actions, rewards, next_states):
        
        outputs = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_outputs = self.model(next_states).detach().max(1)[0]
        targets = rewards + (self.gamma * next_outputs)
        
        loss = torch.nn.functional.smooth_l1_loss(outputs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update(self, new_state, new_reward):
        
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward]), new_state))
        
        new_action = self.get_action(new_state)
        
        if len(self.memory.memory) > 100:
            states, actions, rewards, next_states = self.memory.sample(100)
            self.learn(states, actions, rewards, next_states)
            
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        
        return new_action


# In[ ]:




