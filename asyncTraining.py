#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import *
from trainingScheduler import TrainingScheduler
from agent import Agent
from play import self_play, eval_play
from train import train
import torch
import torch.optim as optim
import ray
import time
import os
import pickle

if __name__ == '__main__':
    
    ray.init()
    
    # initialize scheduler for synchornizing self-play with training/evaluation 
    scheduler = TrainingScheduler.remote()
    
    # if continuing, get replay buffer
    if os.path.isdir('./self_play_data/replay_buffer.pickle'):
        with open('./self_play_data/replay_buffer.pickle', 'rb') as file:
            buffer = pickle.load(file)
        scheduler.add_training_data.remote(list(buffer))
    
    # initialize apprentice agent and alpha agent for training/evaluation
    apprentice_agent = Agent(apprentice=True)
    alpha_agent = Agent(alpha=True)
    alpha_agent.model.eval()   
    
    # initialize SGD optimizer for training, if continuing, load parameters
    if os.path.isdir('./model_data/SGD.pt'):
        optimizer = optim.SGD()
        optimizer.load_state_dict(torch.load('./model_data/SGD.pt'))
    else:
        optimizer = optim.SGD(apprentice_agent.model.parameters(),
                              lr=LEARN_RATE,
                              momentum=MOMENTUM)

    # if needed create directory for pickled self-play data    
    if not os.path.isdir('./self_play_data'):
        os.mkdir('./self_play_data')
        

    # start parallel self-play 
    for process_id in range(ASYNC_SELF_PLAYS):
        self_play.remote(scheduler,
                         process_id)
        
    # wait until replay buffer has sufficient amount of data 
    time.sleep(WAIT)
        
    # start training ---> evaluation loop
    print('Starting training, evaluation loop...')
    while True:
        
        #pickle the replay buffer
        pickle.dump(ray.get(scheduler.get_replay_buffer.remote()), 
                    open('./self_play_data/replay_buffer.pickle' , 'wb' ) )
        
        # train
        print('Training in progress...')
        train(apprentice_agent,
              optimizer,
              scheduler)
        
        # save apprentice model and optimizer parameters
        torch.save(apprentice_agent.model.state_dict(), 
                   './model_data/apprentice_model.pt')
        torch.save(optimizer.state_dict(), './model_data/SGD.pt')
        
        # evaluate
        print('Evaluation in progress...')
        update = eval_play(alpha_agent,
                           apprentice_agent)
                            
        
        # if evaluation went well, update alpha model parameters 
        # and notify self-play agents.
        if update:
            print('The alpha is being updated...')
            torch.save(apprentice_agent.model.state_dict(),
                       './model_data/alpha_model.pt')
            alpha_agent.model.load_state_dict(torch.load(
                './model_data/alpha_model.pt'))
            scheduler.send_signal.remote()
        
        
        
        