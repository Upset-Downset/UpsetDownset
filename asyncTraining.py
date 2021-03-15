#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import *
from trainingScheduler import TrainingScheduler
from agent import Agent
from play import self_play, eval_play
from train import train
import ray
import os
import pickle
import time

if __name__ == '__main__':
    
    ray.init()
    
    ###########################################################################
    ###########################################################################
    
    # initialize scheduler for synchornizing self-play with training/evaluation 
    scheduler = TrainingScheduler.remote()
        
    ###########################################################################
    ###########################################################################
    
    # if havent initialized model yet
    if not os.path.isdir('./model_data'):
        Agent.initialize()
        
    # if needed create directory for pickled self-play data    
    if not os.path.isdir('./self_play_data'):
        os.mkdir('./self_play_data')
    
    # start parallel self-play 
    for process_id in range(ASYNC_SELF_PLAYS):
        self_play.remote(scheduler,
                         process_id)
        
    # wait until replay buffer has sufficient amount of data
    print('Waiting to start training...')
    time.sleep(WAIT_TO_TRAIN)
    
    ###########################################################################
    ###########################################################################
      
    # start training
    print('Starting training...')
    train.remote(scheduler)
    
    # wait until replay buffer has sufficient amount of data
    print('Waiting to start evaluation...')
    time.sleep(WAIT_TO_EVAL)
    
    ###########################################################################
    ###########################################################################
    
    # start evaluation
    print('Starting evaluation...')
    eval_play.remote(scheduler)
        
        
                            

        
        
        
        