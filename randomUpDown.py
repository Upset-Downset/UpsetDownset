#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:40:51 2020

@author: charlie
"""
from poset import *
from upDown import *
import random
import dagUtility as dag

class randomPoset(Poset):
    ''' Subclass of Poset for randomly generated red-green-bue colored posets. 
    Not to be accessed directly, but via the randomGame class.
    '''
    def __init__(self, n, coloring):
        ''' Initializes a randomly generated poset of cardinality 'n' w/ 
        'coloring'.
    
        Parameters
        ----------
        n : int
            The cardinality of the poset. 
        coloring : str
             'random' for a randomly generated coloring and 'all green'  for 
             an all green coloring. 
            
        Return
        -------
        None

        '''
        relations = {i:[] for i in range(n)}
        # Randomy choose the number of edges to be between0 and n(n-1)/2.
        # this is the maximum number of edges possible in a dag having n nodes.
        e = random.randint(0, n*(n-1)/2)
        #add edges. 
        while e > 0:
            #pick two distinct nodes randomly
            i = random.randint(0, n-1)
            j = i
            while j == i:
                j = random.randint(0, n-1)
            # add edge from i to j
            relations[i].append(j)
            #check for cycles, if none count edge as used otherwise get rid of last 
            # edge and start over.
            if dag.isAcyclic(relations):
                e -= 1
            else:
                relations[i].pop()
        # Transitively reduce.
        relations = dag.transitiveReduction(relations)
        # color elements
        coloring = dag.nodeColoring(relations, coloring)
        Poset.__init__(self, relations, coloring, cover = True)

class RandomGame(UpDown):
    ''' Subclass of Updaown for games of upset-downset on randomly generated 
    red-blue-green posets.
    '''
    def __init__(self, n, coloring = 'all green'):
        ''' Initializes a game of upset-downset on a randomly generated poset 
        of cardinality 'n'. Optionally colored by 'random' of 'all green' by default.
        
        Parameters
        ----------
        n : int (nonnegative)
            cardinality of the underlying poset.
        coloring : str, optional
            coloring of unerlying poset. If 'random' the elements of the 
            underlying poset will be colored randomly. Otherwise, all elements 
            will be colored green.

        Returns
        -------
        None

        '''
        poset = randomPoset(n, coloring)   
        UpDown.__init__(self, poset)