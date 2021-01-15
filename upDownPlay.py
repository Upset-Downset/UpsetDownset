#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:16:36 2020

@author: jamison

Play a randomized version of UpDown through the terminal.
"""
import os
import sys
mycwd = os.getcwd()
os.chdir('..')
prevwd = os.getcwd()
sys.path.append(prevwd)
os.chdir(mycwd)

import matplotlib.pyplot as plt
from randomUpDown import RandomGame
import agents

#set marker
#set continuation token
#set same game token
marker = 'o'
cont = 'y'
same_game = 'n'
agent_1=None
agent_2=None
print("\n\n\n\n---------------------------\n---------------------------\n---------------------------\n---------------------------\n-------Upset-Downset-------\n---------------------------\n---------------------------\n---------------------------\n---------------------------\n\n\n\n")
#keep playing games if continuation token is on "y".
while cont == 'y':
    
    #clean up the plots!
    plt.close("all")
    
    #if not playing on a previous game, choose the number of node and color options
    if same_game != 'y':
        while True:
            try:            
                n = abs(int(input("Number of vertices: ")))
                break
            except ValueError:
                print("Please choose an integer.")
        while True:
            c = input("Colors? [y/n]: ")
            if c == 'upsetupsetdownsetdownsetleftsetrightsetleftsetrightsetbsetasetstartset':
                vs = input("Choose vertex style. [ball/star]: ")
                if vs == 'star':
                    marker = "*"
            if c in {'y','n'}:
                break
            else:
                print("Please choose a valid option.")
    
    #set colored to desired option relative to users' choice.
    if c == 'y':
        colored = True
    else:
        colored = False
        
    #initialize a new game if not playing on a previous one.
    if same_game != 'y':
    	G = RandomGame(n, colored=colored)
    
    play_with_agent = input("Play against the computer? [y/n]: ")
    if play_with_agent == 'y':
        agent_order = input("Computer is player 1 [1], player 2 [2], or both[b]?: ")
        if agent_order == '1':
            agent_1 = agents.random_agent
            agent_2 = None
        if agent_order == '2':
            agent_2 = agents.random_agent
            agent_1 = None
        if agent_order == 'b':
            agent_2 = agents.random_agent
            agent_1 = agents.random_agent
        
    #play the game
    G.play(marker=marker, agent_1=agent_1, agent_2=agent_2)
    
    #do you want to keep playing?
    cont = input("Play again? [y/n]: ")
    
    #If you want to keep playing, do you want to keep playing on the same game?
    if cont == 'y':
        same_game = input("Same game? [y/n]: ")
