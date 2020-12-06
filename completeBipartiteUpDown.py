#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti 
"""

import upDown as ud

def bipartite_relations(graphs):
    ''' Returns cover relations for posets whose Hasse diagram is a disjoint 
    union of (horizontally-oriented) complete bipartite graphs.
    
    Parameters
    ----------
    graphs : list
        ordered pairs (tuples or lists) of non-negative integers ('m','n') each 
        representinting a distinct (horizontally-oriented) complete bipartite 
        graph having 'm' nodes on top and 'n' nodes on bottom.
    Returns
    -------
    dict
        xy-coordinates (tuple) of nodes in the Hasse diagram of the poset 
        keyed by corresponding elements.

    '''
    cover_relations = {}
    # variables to store the total number of elements we've processed
    last_count = 0  
    for graph in graphs:
        ##### determine relations for current graph ####
        graph_rels = {}
        # number of elements on bottom and top.
        bottom_count = graph[1]
        top_count = graph[0]
        # keep track of total number of elements in this graph.
        count = top_count + bottom_count
        # label elements on top and bottom.
        bot_elems = [i for i in range(last_count, last_count + bottom_count)]
        top_elems = [i for i in range(last_count + bottom_count, last_count + count)]
        # relations amongst bottom and top elements
        graph_rels.update({i: top_elems for i in bot_elems})
        graph_rels.update({i:[] for i in top_elems})
        # updates before processing next graph
        cover_relations.update(graph_rels)      
        last_count += count
    return cover_relations

def bipartite_coordinates(graphs):
    ''' Returns Hasse diagram node coordinates for posets whose Hasse diagram 
    is a disjoint union of (horizontally-oriented) complete bipartite graphs.
    
    Parameters
    ----------
    graphs : list
        ordered pairs (tuples or lists) of non-negative integers ('m','n') each 
        representinting a distinct (horizontally-oriented) complete bipartite 
        having 'm' nodes on top and 'n' nodes on bottom.
    Returns
    -------
    dict
        keyed by elements of the poset (nonnegative integers) with 
        corresponding value being the xy-coordinate (tuple) of the
        corresponding node in the Hasse diagram.

    '''
    coordinates = {}
    # variables to store the total number of elements we've processed and the
    # largest x-coordinate we've assigned as we loop over the graphs
    last_count = 0
    last_x = 0
    for graph in graphs:
        graph_coords = {}
        # number of elements on bottom and top.
        bottom_count = graph[1]
        top_count = graph[0]
        # keep track of total number of elements in this graph.
        count = top_count + bottom_count
        ##### determine positions for current graph ####
        # keep track of x-coordinates
        current_x = last_x
        # at least as many nodes on top
        if top_count >= bottom_count:
            for i in range(last_count + bottom_count, last_count + count):
                graph_coords[i] = (current_x, 1)
                current_x += 1
            start_x = last_x 
            last_x = current_x - 1
            top_center_x = (last_x + start_x) / 2
            half_bottom_count = bottom_count // 2
            bottom_x = top_center_x - half_bottom_count - ((bottom_count % 2) -1) / 2
            for i in range(last_count, last_count + bottom_count):
                graph_coords[i] = (bottom_x, 0)
                bottom_x += 1
        # more nodes on bottom
        else:
            for i in range(last_count, last_count + bottom_count):
                graph_coords[i] = (current_x, 0)
                current_x += 1
            start_x = last_x 
            last_x = current_x - 1
            bottom_center_x = (last_x + start_x) / 2
            half_top_count = top_count // 2
            top_x = bottom_center_x - half_top_count - ((top_count % 2) -1) / 2
            for i in range(last_count + bottom_count, last_count + count):
                graph_coords[i] = (top_x, 1)
                top_x += 1
        # updates before processing next graph
        coordinates.update(graph_coords)
        last_x += 2       
        last_count += count
    return coordinates

class KGame(ud.UpDown):
    ''' Subclass of UpDown for games of upset-downset on posets whose Hasse 
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
            having m nodes on top and n nodes on bottom. 

        Returns
        -------
        None

        '''
        coloring = None
        covers = bipartite_relations(graphs)
        coordinates = bipartite_coordinates(graphs)
        ud.UpDown.__init__(self, covers, coloring, coordinates = coordinates,\
                           covers = True)
        # Set graphs attribute
        self.graphs = graphs
                        
    #def standard_form(self):
        
    #def symmetric_options(self):
        
    #def outcome(self): using symmetric_options() and standard_form()
            
            
