#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:39:08 2020
@author: Charles Petersen and Jamison Barsotti
"""
import os
import sys
mycwd = os.getcwd()
os.chdir('..')
prevwd = os.getcwd()
sys.path.append(prevwd)
os.chdir(mycwd)


import numpy as np



class game_state(object):
    
    def __init__(self, game, n = 10):
        mat = np.zeros((n, n))
        G = game.cover_relations
        colors = game.coloring
        
        # add upper cover relations: 
        # if  node i <  node j, set mat[i,j] == 1 in poset underlying the game
        for node in G:
            for cover in G[node]:
                mat[node][cover] = 1
                
        # add lower cover relations:
        # if node i > node j mat[i,j] == -1
        t = np.transpose(mat)
        mat = mat - t
        
        # add color info:
        # if node i has color c, set mat[i,i] == c + 2
        for color in colors:
            mat[color][color] = colors[color]+2
        self.universe_size = n
        self.matrix = mat
            
    def update_state(self, features_removed):
        
        n = self.universe_size
        zero_array = np.zeros((n,))
        for index in features_removed:
            self.matrix[index,:] = zero_array
            self.matrix[:,index] = zero_array
       
