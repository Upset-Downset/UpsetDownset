#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti

NOTE: Though the paper 'Mastering the game of Go without Human Knowledge' 
by David Silver Et Al. does a great job explaining the architecture of 
the model, the blog https://dylandjian.github.io/alphago-zero/ helped us 
understand the syntax needed to implement the model in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# game state variables
STATE_SHAPE = (4,10,10) # the shape of an encoded game state
NUM_ACTIONS = 10        # the number of 'possible' actions available

# network architecture variables
NUM_FILTERS = 64    # the number of filters in the input and residual blocks
NUM_RES_BLOCKS = 10 # the number of residual blocks
NUM_POL_FILTERS = 2 # the number of filters in the policy head
NUM_VAL_FILTERS = 1 # the number of filters in the value head
NUM_HIDDEN_VAL = 64 # the size of the hidden layer in the value head

# The first layer in the network architecture is a single convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, board_shape):       
        super(ConvBlock, self).__init__()      
        self.in_channels = in_channels
        self.board_shape = board_shape        
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):       
        # torch expects a batch
        x = x.view(-1, 
                   self.in_channels, 
                   self.board_shape[0], 
                   self.board_shape[1])    
        
        # Convolution-->Batch Norm-->Leaky Relu
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)  
        
        return x
    
# The second layer in the network architecture is composed
# of many residual blocks in sequence
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                              out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):      
        # the skip connection
        skip = x
        
        # Convolution-->Batch Norm-->Leaky Relu
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.leaky_relu(out)
        
        # Convolution-->Batch Norm-->Leaky Relu
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.leaky_relu(out)
        
        # skip connection-->Leaky Relu
        out += skip
        out = F.leaky_relu(out)
        
        return out

# The final layer in the network architecture is split between 
# a policy head and a value head
class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_channels, board_shape, num_actions):
        super(PolicyHead, self).__init__()
        self.mix_channels = out_channels*(board_shape[0]*board_shape[1])  
        self.conv = nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size=1, 
                              stride=1 )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dense = nn.Linear(self.mix_channels, num_actions)
        
    def forward(self, x): 
        #Convolution-->Batch Norm-->Leaky Relu
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        
        # Dense layer-->Log Softmax
        x = x.view(-1, self.mix_channels)
        x = self.dense(x)
        policy = self.log_softmax(x).exp() 
        
        return policy
    
class ValueHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_layer_size, 
                 board_shape):
        super(ValueHead, self).__init__()
        self.mix_channels = out_channels*(board_shape[0]*board_shape[1])
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.dense1 = nn.Linear(self.mix_channels, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, 1)
        
    def forward(self, x):
        # Convolution-->Batch Norm--> Leaky Relu
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)
        
        # Linear-->Leaky Relu
        x = x.view(-1, self.mix_channels)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        
        # Linear-->Tanh
        x = self.dense2(x)
        value = torch.tanh(x)
        
        return value
    
# the full network architecture     
class AlphaZeroNet(nn.Module):
    def __init__(self,
                 in_channels=STATE_SHAPE[0],
                 board_shape=STATE_SHAPE[1:],
                 num_actions=NUM_ACTIONS,
                 num_res_blocks=NUM_RES_BLOCKS,
                 num_filters=NUM_FILTERS,
                 num_pol_filters=NUM_POL_FILTERS,
                 num_val_filters=NUM_VAL_FILTERS,
                 num_hidden_val=NUM_HIDDEN_VAL):
        
        super(AlphaZeroNet, self).__init__()
        self.num_res_blocks = num_res_blocks
        # convolutional block
        self.conv = ConvBlock(in_channels, num_filters, board_shape)
        # residual blocks
        for block in range(self.num_res_blocks):
            setattr(self, f'residual_{block}', 
                    ResidualBlock(num_filters, num_filters))
        # policy head
        self.policy_head = PolicyHead(num_filters, 
                                      num_pol_filters, 
                                      board_shape,
                                      num_actions)
        # value head
        self.value_head = ValueHead(num_filters, 
                                    num_val_filters, 
                                    num_hidden_val,
                                    board_shape)
        
    def forward(self, x):       
        # Convolution-->Residuals-->Policy/Value
        x = self.conv(x)
        for block in range(self.num_res_blocks):
            x = getattr(self, f'residual_{block}')(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
        
    
    
        
        
        
        
    
    