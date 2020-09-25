#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:16:36 2020

@author: jamison
"""
import upDown as uD
import matplotlib.pyplot as plt

cont = 'y'
same_game = 'n'
while cont == 'y':
    plt.close("all")
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
    
    if c == 'y':
        colored = True
    else:
        colored = False
    if same_game != 'y':
    	G = uD.RandomUpDown(n, colored=colored)
    G.play(marker=marker)
    cont = input("Play again? [y/n]: ")
    if cont == 'y':
        same_game = input("Same game? [y/n]: ")
