#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""
import os
import sys
mycwd = os.getcwd()
os.chdir('..')
prevwd = os.getcwd()
sys.path.append(prevwd)
os.chdir(mycwd)


import random
def random_agent(game, player):
    ''' Returns a random move in the upset-downset 'game' for 'player'.
    Parameters
    ----------
    game : UpDown
        a game of upset-downset
    player : str
        'Up' if the agent is to be the Up player, and 'Down' if the agent is
        to be the Down player.

    Returns
    -------
    int
        element in the poset underlying the upset-downset 'game' on which 
        the agent is to make its random play.

    '''
    # Determine which colors agent chooses.
    if player == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0}   
    # Find the list of opions for the agent
    options = list(filter(lambda x : game.coloring[x] in \
                          colors, list(game.elements)))
    # Return a random choice.    
    return random.choice(options)


def min_agent(game, player):
    ''' Returns a move in the upset-downset 'game' for 'player' which removes 
    the least elements from the board.
    
    Parameters
    ----------
    game : UpDown
        a game of upset-downset
    player : str
        'Up' if the agent is to be the Up player, and 'Down' if the agent is
        to be the Down player.

    Returns
    -------
    int
        element in the poset underlying the upset-downset 'game' on which 
        the agent is to make its minimum play.

    '''
    # Determine which colors agent chooses.
    if player == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0}    
    # Find the list of opions for the agent
    options = list(filter(lambda x : game.coloring[x] in \
                          colors, list(game.elements)))
    # pick option which removes the least elements from the board
    u = options[0]
    if player == 'Up':
        size = len(game.upset(u))
        for x in options[1:]:
            size_x = len(game.upset(x))
            if size > size_x:
                u = x
                size = size_x
    else:
        size = len(game.downset(u))
        for x in options[1:]:
            size_x = len(game.downset(x))
            if size > size_x:
                u = x
                size = size_x
    return u

def max_agent(game, player): 
    ''' Returns a move in the upset-downset 'game' for 'player' which removes 
    the most elements from the board.
    
    Parameters
    ----------
    game : UpDown
        a game of upset-downset
    player : str
        'Up' if the agent is to be the Up player, and 'Down' if the agent is
        to be the Down player.

    Returns
    -------
    int
        element in the poset underlying the upset-downset 'game' on which 
        the agent is to make its maximum play.

    '''
    # Determine which colors agent chooses.
    if player == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0} 
    # Find the list of opions for the agent
    options = list(filter(lambda x : game.coloring[x] in \
                          colors, list(game.elements)))
    # pick option which removes the most elements from the board
    u = options[0]
    if player == 'Up':
        size = len(game.upset(u))
        for x in options[1:]:
            size_x = len(game.upset(x))
            if size < size_x:
                u = x
                size = size_x
    else:
        size = len(game.downset(u))
        for x in options[1:]:
            size_x = len(game.downset(x))
            if size < size_x:
                u = x
                size = size_x
    return u
       
        
    