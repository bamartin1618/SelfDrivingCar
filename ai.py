#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os 
import random

# Import PyTorch libraries to implement the DQN.
import torch
import torch.nn
import torch.optim
import torch.nn.functional
from torch.autograd import Variable

class Network(torch.nn.Module): # Class that builds a vanilla ANN in PyTorch. For our implementation, we want to inherit the existing Module class from PyTorch.
    
    def __init__(self, input_size, nb_action): # Takes as parameters the number of neurons in the input layer and number of neurons in the output layer.
        
        super(Network, self).__init__() # Allows us to inherit the properties and methods of the Module class.
        
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = torch.nn.Linear(input_size, 30) # Forms the connections between input layer and the hidden layer. In our ANN, all hidden layers have thirty neurons.
        self.fc2 = torch.nn.Linear(30, nb_action) # Forms the connections between the hidden layer and the output layer.
        
    def forward(self, state):
        
        x = torch.nn.functional.relu(self.fc1(state)) # Forward propagates the inputs (the state) in the input layer to the hidden layer, which activates the neurons in the hidden layer using the ReLU activation function.
        
        Q_values = self.fc2(x) # Forward propagates the activations in the hidden layer neurons to the output layer with no activation function, giving us the Q values.
        
        return Q_values
    
class ReplayMemory(object): # Class to store memory for the agent, intended for long-term learning.
    
    def __init__(self, capacity): # Takes as a parameter the capacity of the agent's memory.
        
        self.capacity = capacity
        self.memory = []
        
    def push(self, event): # Pushes an event (previous state vector, reward, action, current state vector) to the agent's memory.
        
        self.memory.append(event) # Adds the event to the agent's memory.
        
        if len(self.memory) > self.capacity:
            del self.memory[0] # We treat the agent's memory as a queue. If it becomes too large (over capacity), we remove events added earliest first.
            
    def sample(self, batch_size):
        
        """
        Generate a random sample of events (length is determined by the batch size) from the agent's memory.
        The zip function regroups the data (states, actions, rewards) into a format that is readable by PyTorch.
        """
        
        samples = zip(* random.sample(self.memory, batch_size))
  
        
        """
        Convert each event to a Variable object, which associates a tensor with a gradient, allowing
        for faster computation when we update the weights of the network.
        """
        
        return map(lambda x: Variable(torch.cat(x, 0)), samples) 
   
    
class DQN(object): # Class for the DQN - the brain of the agent.
    
    def __init__(self, input_size, nb_action, gamma): # Takes as parameters the number of inputs, the number of actions, and the learning rate.
        
        """
        Initialize attributes of the DQN, such as the underlying ANN, the memory of the agent, the ANN's optimizer, etc.
        """
        
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = torch.optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        self.last_action = 0
        self.last_reward = 0
        
    def get_action(self, state): # Compute the best action for a particular state.
        
        probabilities = torch.nn.functional.softmax(self.model(Variable(state)) * 100) # Pass Q-values predicted by the ANN to a softmax function.
        action = probabilities.multinomial(len(probabilities)) # Compute the predicted action using softmax.
        
        return action.data[0, 0]
    
    def learn(self, states, actions, rewards, next_states): # Optimizes the ANN given a batch of sampled events.
        
        outputs = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Compute the Q-values of the states.
        
        next_outputs = self.model(next_states).detach().max(1)[0] # Compute the maximum Q-values from the next_states.
        
        targets = rewards + (self.gamma * next_outputs) # Compute the target Q-values.
        
        loss = torch.nn.functional.smooth_l1_loss(outputs, targets) # Compute the loss between the output and target Q-values.
        
        # Update the weights of the DQN through backpropagation using the optimizer.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update(self, new_state, new_reward): # Updates the memory of the agent and lets the agent learn by updating the DQN.
        
        new_state = torch.Tensor(new_state).float().unsqueeze(0) # Turns the state into a tensor so it can be fed to the DQN
        
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward]), new_state)) # Push the state to the memory of the DQN.
        
        new_action = self.get_action(new_state) # Compute the prediction action. 
        
        if len(self.memory.memory) > 100:
            states, actions, rewards, next_states = self.memory.sample(100) # Take a random sample of 100 events.
            self.learn(states, actions, rewards, next_states) # Update the weights of the DQN.
            
        # Update the previous state, action, and reward.
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        
        return new_action


# In[ ]:




