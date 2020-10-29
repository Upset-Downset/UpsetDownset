#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:27:38 2020

@author: jamison
"""



def random_agent(upDown, set_type):
    
    import random
    
    # Determine which colors agent chooses.
    if set_type == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0}
    
    # Find the list of opions for the agent
    options = list(filter(lambda x : upDown._coloring_map[x] in colors, \
                  list(upDown._cover_relations.keys())))

    # Return a random choice.    
    return random.choice(options)


def min_agent(upDown, set_type):
    
    # Determine which colors agent chooses.
    if set_type == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0}
    
    # Find the list of opions for the agent
    options = list(filter(lambda x : upDown._coloring_map[x] in colors, \
                  list(upDown._cover_relations.keys())))
    u = options[0]
    if set_type == 'Up':
        size = len(upDown.upset(u))
        for x in options[1:]:
            size_x = len(upDown.upset(x))
            if size > size_x:
                u = x
                size = size_x
    else:
        size = len(upDown.downset(u))
        for x in options[1:]:
            size_x = len(upDown.downset(x))
            if size > size_x:
                u = x
                size = size_x
    return u

def max_agent(upDown, set_type):
    
    # Determine which colors agent chooses.
    if set_type == 'Up':
        colors = {0,1}
    else:
        colors = {-1,0}
    
    # Find the list of opions for the agent
    options = list(filter(lambda x : upDown._coloring_map[x] in colors, \
                  list(upDown._cover_relations.keys())))
    u = options[0]
    if set_type == 'Up':
        size = len(upDown.upset(u))
        for x in options[1:]:
            size_x = len(upDown.upset(x))
            if size < size_x:
                u = x
                size = size_x
    else:
        size = len(upDown.downset(u))
        for x in options[1:]:
            size_x = len(upDown.downset(x))
            if size < size_x:
                u = x
                size = size_x
    return u
       
        
    