import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainMLP import GRU

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_actions):
        super(Actor, self).__init__()

        # Make a simple 3 later linear network
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_actions = max_actions

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.max_actions * torch.tanh(self.l3(x))
        return x

class DeeSpineActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_actions):
        super(Actor, self).__init__()

        # Make a simple 3 later linear network
        self.n_ees = 19
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, self.n_ees)
        self.max_actions = max_actions

        # Intialize, load, and freeze the DeepSpine model
        self.deepspine = GRU(self.n_ees, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.deepspine.forward(x)
        x = self.max_actions * torch.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Defined the Q1 and Q2 of the TD3.
        # https://arxiv.org/pdf/1802.09477.pdf
        super(Critic, self).__init__()
        # Q1. Final layer of Q1 to return single value.
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1) 
        
        # Q2. Same as Q1. 
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1) 
        
    def forward(self, state, action): 
        # Perform forward pass through NN with the given state
        # and the action to take on this state.
        # Flatten state + action so we have the input value shape to 
        # pass to the NN. 
        sa = torch.cat([state, action], 1) 

        # Q1 value computation. 
        c1 = F.relu(self.l1(sa))
        c1 = F.relu(self.l2(c1))
        c1 = self.l3(c1)

        # Q2 value computation.
        c2 = F.relu(self.l4(sa))
        c2 = F.relu(self.l5(c2))
        c2 = self.l6(c2)

        # Return both values so we can grab the min of the two. 
        return (c1, c2)

