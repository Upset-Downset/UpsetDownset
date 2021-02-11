#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: charlie
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--main_loop_iters', type=int, default=10,help='Number of iterations of self-play, train, evaluation loop')
parser.add_argument('--num_processes', type=int, default=4, help='Number of self-play and evaluation processes')
parser.add_argument('--num_self_plays', type=int, default=100, help='Number of games to play during each round of self-play')
parser.add_argument('--num_mcts_searches', type=int, default=400, help='Number of iterations of search to perform during self-play and evaluation mcts')
parser.add_argument('--temp', type=int, default=1, help='Exponent for stochastic mcts policy evaluation')
parser.add_argument('--temp_threshold', type=int, default=2, help='Number of opening moves after which mcts policy becomes deterministic')
parser.add_argument('--training_batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--num_symmetries', type=int, default=10, help='Number of symmetries to obtain for each tarining example')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for SGD')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--num_eval_plays', type=int, default=400, help='Number of games to play during each round of evaluation')
parser.add_argument('--win_ratio', type=float, default=0.55, help='Ratio of wins the apprentice model must have to be considered better than the alpha')
parser.add_argument('--eta', type=float, default=1.0, help='Parameter for Dirichelt distribution')
parser.add_argument('--eps', type=float, default=0.25, help='Parameter controling ratio of Dirichlet noise in PUCT edge selection policy')
parser.add_argument('--max_nodes', type=int, default=10, help='Maximum number of nodes the model can play on')
parser.add_argument('--num_residual_blocks', type=int, default=5, help='Number of residual blosk in model architecture')
parser.add_argument('--num_filters', type=int, default=64, help='Number of filters in each convolution')
