#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:59:39 2021

@author: jamison
"""
import os
import random
import argparse
import collections
import numpy as np

import gameState as gs 
import model, mcts
import randomUpDown as rud

#from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F


PLAY_EPISODES = 1  #25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 5000 # 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000 #10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 100
STEPS_BEFORE_TAU_0 = 10

TEMP = 1
TEMP_THRESH = 3


def evaluation(alpha_net, apprentice_net, num_plays=400, win_thrshld=0.55, temp = 0):
    '''
    SIMILAR TO self_play BUT USES TWO MCTSes AND AN INFINTESIMAL TEMP ALWAYS
    '''
    alpha = 0
    apprentice = 1
    apprentice_wins = 0
    actions = np.arange(gs.UNIV)
    
    for i in range(num_plays):
        print(i)
        net_to_start = np.random.choice([alpha, apprentice])
        up_or_down = np.random.choice([gs.UP, gs.DOWN])
        G = rud.RandomGame(gs.UNIV, colored=True)
        cur_state = gs.to_state(G, first_move = up_or_down)
        cur_net = net_to_start
        move_count = 0
        winner = None
        while not gs.is_terminal_state(cur_state):
            root = mcts.PUCTNode(cur_state)
            if cur_net == alpha:
                mcts.MCTS(root, alpha_net)
            else:
                mcts.MCTS(root, apprentice_net)
            policy = mcts.MCTS_policy(root, temp)
            move = np.random.choice(actions, p=policy)
            cur_state = root.edges[move].state
            cur_net = 1 - cur_net
            move_count += 1
        if move_count %2 == 0:
            winner = 1-net_to_start
        else:
            winner = net_to_start
        if winner == apprentice:
            apprentice_wins+=1
    if (apprentice_wins/num_plays) > win_thrshld:
        return True
    else: 
        return False
        


if __name__ == "__main__":

    apprentice_net = model.Net(input_shape=model.OBS_SHAPE, actions_n=gs.UNIV)
    torch.save(apprentice_net.state_dict(), '0alpha_net.pt')
    #print(apprentice_net)

    optimizer = optim.SGD(apprentice_net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    
    
    step_idx = 0
    best_idx = 0

    while True:
        
        for _ in range(PLAY_EPISODES):
            size = random.randint(1, gs.UNIV)
            G = rud.RandomGame(size, colored=True)
            first_move = random.choice(['UP','DOWN'])
            initial_state = gs.to_state(G, dim=gs.UNIV, first_move=first_move)
            train_data = mcts.self_play(initial_state, apprentice_net, temp=TEMP, tmp_thrshld=TEMP_THRESH)
            list(map(replay_buffer.append, train_data))
        
        step_idx += 1

        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        # train
        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0

        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_probs, batch_values = zip(*batch)
            
            

            optimizer.zero_grad()
            states_v = torch.FloatTensor(batch_states)
            probs_v = torch.FloatTensor(batch_probs)
            values_v = torch.FloatTensor(batch_values)
            out_logits_v, out_values_v = apprentice_net(states_v)

            loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
            loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
            sum_loss += loss_v.item()
            sum_value_loss += loss_value_v.item()
            sum_policy_loss += loss_policy_v.item()

        # evaluate net
        if step_idx % EVALUATE_EVERY_STEP == 0:
            alpha_net = model.Net(input_shape=model.OBS_SHAPE, actions_n=gs.UNIV)
            alpha_net.load_state_dict(torch.load(str(best_idx) + 'alpha_net.pt'))
            apprentice_better_than_alpha = evaluation(apprentice_net, alpha_net, num_plays=EVALUATION_ROUNDS, temp=TEMP)

            if apprentice_better_than_alpha:
                print("Net is better than cur best, sync")
                best_idx += 1
                s = str(best_idx)
                torch.save(apprentice_net.state_dict(), s + 'alpha_net.pt')
                #mcts_store.clear()

