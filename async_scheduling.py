"""
@author: charlie
"""
from config import *
import numpy as np
from collections import deque
import random
import asyncio
import ray

@ray.remote(num_cpus=0)
class AsyncSignal(object):
    
    def __init__(self, clear=False):
        self.training_signal = asyncio.Event()
        self.clear = clear

    def send(self):
        self.training_signal.set()
        if self.clear:
            self.training_signal.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.training_signal.wait()

@ray.remote(num_cpus=0)            
class UpdateSignal(object):
    def __init__(self, size=ASYNC_SELF_PLAYS):
        self.signal = np.zeros(size, dtype=bool)
        
    def send_update(self):
        self.signal[:] = True
    
    def get_update(self, process_id):
        return self.signal[process_id]
        
    def confirm_update(self, process_id):
        self.signal[process_id] = False
        
@ray.remote(num_cpus=0)
class ReplayBuffer(object):
    
    def __init__(self, training_signal, size=MAX_REPLAY_BUFFER):
        self.buffer = deque(maxlen=size)
        self.training_signal = training_signal
        self.total_games = 0
   
    def get_total_games(self):
        return self.total_games
    
    def get_total_positions(self):
        return len(self.buffer)
    
    def add(self, data, games_to_train=GAMES_TO_TRAIN):
        self.buffer.extend(data)
        self.total_games += 1
        if self.total_games == games_to_train:
            self.training_signal.send.remote()
    
    def get_batch(self, batch_size=BATCH_SIZE):
        return random.choices(self.buffer, k=batch_size)