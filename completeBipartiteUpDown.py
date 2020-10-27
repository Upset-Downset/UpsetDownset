#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:02:41 2020

@author: charlie
"""

from upDown import *
import dagUtility as dag

def complete_bipartite_union(graphs):
    ''' Returns cover relations and Hasse diagram node positions
    for posets whose Hasse diagram is a disjoint union of (horizontally 
    -oriented) complete bipartite graphs.
    
    Parameters
    ----------
    graphs : list
        ordered pairs (tuples or lists) of non-negative integers (m,n) each 
        representinting a distinct (horizontally-oriented) complete bipartite 
        having m nodes on top and n nodes on bottom.
    Returns
    -------
    dict
        of dicts: dict of poset relations keyed by 'relations', and dict of the
        corresponding Hasse diagram node positions keyed by 'positions'.     
            - 'relations' dict is keyed by elements of the poset (consecutive 
            nonnegative integers starting at 0) with corresponding value being 
            the list of (upper) covers for that element.
            - 'positions' dict is keyed by elements of the poset (consecutive 
            nonnegative integers starting at 0) with corresponding value being 
            the tuple containg the xy-position of that element in the Hasse 
            diagram of the poset.

    '''
    cover_relations = {}
    bipartite_positions = {}
    complete_bipartite = {'relations':cover_relations, 'positions':bipartite_positions}
    # variables to store the total number of elements we've processed and the
    # largest x-coordinate we've assigned as we loop over the graphs
    last_count = 0  
    last_x = 0
    for graph in graphs:
        ##### determine relations for current graph ####
        relations = {}
        positions = {}
        # number of elements on bottom and top.
        bottom_count = graph[1]
        top_count = graph[0]
        # keep track of total number of elements in this graph.
        count = top_count + bottom_count
        # label elements on top and bottom.
        bot_elems = [i for i in range(last_count, last_count + bottom_count)]
        top_elems = [i for i in range(last_count + bottom_count, last_count + count)]
        # relations amongst bottom and top elements
        relations.update({i: top_elems for i in bot_elems})
        relations.update({i:[] for i in top_elems})
        ##### determine positions for current graph ####
        # keep track of x-coordinates
        current_x = last_x
        # at least as many nodes on top
        if top_count >= bottom_count:
            for i in range(last_count + bottom_count, last_count + count):
                positions[i] = (current_x, 2)
                current_x += 1
            start_x = last_x 
            last_x = current_x - 1
            top_center_x = (last_x + start_x) / 2
            half_bottom_count = bottom_count // 2
            bottom_x = top_center_x - half_bottom_count - ((bottom_count % 2) -1) / 2
            for i in range(last_count, last_count + bottom_count):
                positions[i] = (bottom_x, 1)
                bottom_x += 1
        # more nodes on bottom
        else:
            for i in range(last_count, last_count + bottom_count):
                positions[i] = (current_x, 1)
                current_x += 1
            start_x = last_x 
            last_x = current_x - 1
            bottom_center_x = (last_x + start_x) / 2
            half_top_count = top_count // 2
            top_x = bottom_center_x - half_top_count - ((top_count % 2) -1) / 2
            for i in range(last_count + bottom_count, last_count + count):
                positions[i] = (top_x, 2)
                top_x += 1
        # updates before processing next graph
        cover_relations.update(relations)
        bipartite_positions.update(positions)
        last_x += 2       
        last_count += count
    return complete_bipartite

class KGame(UpDown):
    ''' Subclass of Updaown for games of upset-downset on posets whose Hasse 
    diagram is a disjoint union of (horizontally-oriented) complete bipartite 
    graphs.
    '''
    def __init__(self, graphs):
        ''' Initializes an 'all green' game of upset-downset on a poset whose 
        Hasse diagram is a disjoint union of (horizontally-oriented) complete 
        bipartite graphs.
        
        Parameters
        ----------
        graphs : list
            ordered pairs (tuples or lists) of non-negative integers (m,n) each 
            representinting a distinct (horizontally-oriented) complete bipartite 
            having m nodes on top and n nodes on bottom.. 

        Returns
        -------
        None

        '''
        n = 0
        for graph in graphs:
            n += graph[0] + graph[1]
        coloring = {i:0 for i in range(n)}
        Ks = complete_bipartite_union(graphs) 
        covers = Ks['relations']
        UpDown.__init__(self, covers, coloring, covers = True)
        self._graphs = graphs
        self._positions = Ks['positions']  # USE THIS TO PLOT....
        
        
    #def gameboard(self):
            
            