#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:57:35 2021

@author: charlie
"""
import ray
import numpy as np
import random
from collections import deque

@ray.remote
class TrainingScheduler(object):
    def __init__(self,num_trainers, replay_size):
        self.update_signal = np.zeros(num_trainers, dtype=np.int8)
        self.replay_buffer = deque(maxlen=replay_size)
        self.train_iter = 0
        
    def get_signal(self, process_id):
        return self.update_signal[process_id]
    
    def reset_signal(self, process_id):
        self.update_signal[process_id] = 0

    def send_signal(self):
        self.update_signal[:] = 1
    
    def add_training_data(self, data):
        self.replay_buffer.extend(data)
        
    def get_training_batch(self, batch_size):
        return random.choices(self.replay_buffer, 
                                k=batch_size)
    
    def get_replay_buffer(self):
        return self.replay_buffer
    
