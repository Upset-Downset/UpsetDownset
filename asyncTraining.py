#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    
    #params
    NUM_ASYNC_SELF_PLAYS = 2
    SEARCH_ITERS = 200
    TEMP = 1
    TEMP_THRSHLD = 4
    WAIT = 60
    BUFFER_SIZE = 1000000
    NUM_EPOCHS = 10000
    BATCH_SIZE = 4
    NUM_SYMMETRIES = 16
    LEARN_RATE = 0.001
    MOMENTUM = 0.9
    EVAL_PLAYS = 100
    WIN_RATIO = 0.55
    
    # initialize scheduler for synchornizing self-play with training/evaluation 
    scheduler = TrainingScheduler.remote(NUM_ASYNC_SELF_PLAYS, BUFFER_SIZE)
    
    # if continuing, get replay buffer
    if os.path.isdir('./self_play_data/replay_buffer.pickle'):
        buffer = pickle.load(
            open('./self_play_data/replay_buffer.pickle', 'rb'))
        scheduler.add_training_data.remote(list(buffer))
    
    # initialize apprentice agent and alpha agent for training/evaluation
    apprentice_agent = Agent(apprentice=True)
    alpha_agent = Agent(alpha=True)
    alpha_agent.model.eval()   
    
    # initialize SGD optimizer for training
    optimizer = optim.SGD(apprentice_agent.model.parameters(),
                          lr=LEARN_RATE,
                          momentum=MOMENTUM)
    
    # if continuing, load optimizer parameters
    if os.path.isdir('./model_data/SGD.pt'):
        optimizer.load_state_dict(torch.load('./model_data/SGD.pt'))
        

    # start parallel self-play 
    for process_id in range(NUM_ASYNC_SELF_PLAYS):
        self_play.remote(scheduler,
                         process_id,
                         SEARCH_ITERS,
                         TEMP, 
                         TEMP_THRSHLD)
        
    # wiat until replay buffer has sufficient amount of data 
    time.sleep(WAIT)
        
    # start training ---> evaluation loop
    print('Starting training, evaluation loop...')
    while True:
        
        #pickle the replay buffer
        if not os.path.isdir('./self_play_data'):
            os.mkdir('./self_play_data')
        pickle.dump(ray.get(scheduler.get_replay_buffer.remote()), 
                    open('./self_play_data/replay_buffer.pickle' , 'wb' ) )
        
        # train
        print('Training in progress...')
        train(apprentice_agent,
              optimizer,
              scheduler,
              BATCH_SIZE, 
              NUM_SYMMETRIES,
              NUM_EPOCHS)
        
        # save apprentice model and optimizer parameters
        torch.save(apprentice_agent.model.state_dict(), 
                   './model_data/apprentice_model.pt')
        torch.save(optimizer.state_dict(), './model_data/SGD.pt')
        
        # evaluate
        print('Evaluation in progress...')
        update = eval_play(alpha_agent,
                            apprentice_agent,
                            EVAL_PLAYS,
                            SEARCH_ITERS,
                            WIN_RATIO)
        
        # if evaluation went well, update alpha model parameters 
        # and notify self-play agents.
        if update:
            print('The alpha is being updated...')
            torch.save(apprentice_agent.model.state_dict(),
                       './model_data/alpha_model.pt')
            scheduler.send_signal.remote()
        
        
        
        