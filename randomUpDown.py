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
    '''
    Constructor for a game of upset-downset played on a randomly generated poset.
    
    Not to be accessed directly, but via the randomGame class.
    '''
    def __init__(self, n, coloring):
        '''  Initializesa randomly generated poset of cardinality 'n'.
    
        Parameters
        ----------
        n : int
            The cardinality of the poset. 
        coloring : dict
             coloring map: colors keyed by element, where 1 (resp. 0,-1) 
             represent blue (resp. green, red). 
            
        Return
        -------
        Poset object

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
    ''' Constructor for a game of upset-downset played ona randomly generated poset.
    '''
    def __init__(self, n, coloring = 'all green'):
        ''' Initializes a randomly generated game of upset-downset.
        
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
        RandomUpDown object

        '''
        poset = randomPoset(n, coloring)   
        UpDown.__init__(self, poset)