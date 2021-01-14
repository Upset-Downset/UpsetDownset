#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti 
"""

import upDown as ud

def complete_bipartite_dag(graphs):
    ''' Returns a directed acyclic graph corresponding to the complete 
    bipartite graphs, 'graphs'.
    
    Parameters
    ----------
    graphs : list
        ordered pairs of nonnegative integers 'm','n' each 
        representinting a distinct instance of the (horizontally-oriented) 
        complete bipartite  graph having 'm' nodes on top and 'n' nodes on 
        bottom.
        
    Returns
    -------
    dict
        adjacency representation of a directed graph (adjacency lists keyed 
        by node.) corresponding to 'graphs'. A disjoint union of directed 
        acyclic graphs, one for each graph in 'graphs': Each is a directed 
        (horizontally-oriented) complete bipartite graph with edges directed 
        from bottom nodes to top nodes. 

    '''
    dag = {}
    # update total number of nodes processed after each iteration
    last_count = 0  
    for graph in graphs:
        #determine relations for current graph 
        graph_rels = {}
        # number of elements on bottom and top
        bottom_count = graph[1]
        top_count = graph[0]
        # total number of elements in current graph
        count = top_count + bottom_count
        # label elements on top and bottom.
        bot_elems = [i for i in range(last_count, last_count + bottom_count)]
        top_elems = [i for i in range(last_count + bottom_count, last_count + count)]
        # edges amongst bottom and top nodes
        graph_rels.update({i: top_elems for i in bot_elems})
        graph_rels.update({i:[] for i in top_elems})
        # update before processing next graph
        dag.update(graph_rels)      
        last_count += count
    return dag


class CompleteBipartiteGame(ud.UpDown):
    ''' Subclass of UpDown for games of upset-downset on disjoint unions of
    directed (horizontally-oriented) complete bipartite graphs.
    '''
    def __init__(self, graphs, colored = False):
        ''' Initializes an 'all green' game of upset-downset on a poset whose 
        Hasse diagram is a disjoint union of (horizontally-oriented) complete 
        bipartite graphs.
        
        Parameters
        ----------
        graphs : list
            ordered pairs (tuples or lists) of non-negative integers (m,n) each 
            representinting a distinct (horizontally-oriented) complete bipartite 
            having m nodes on top and n nodes on bottom.
       colored : bool, optional
            determines the coloring. If 'True' the nodes will be colored 
            randomly. Otherwise, all nodes will be colored green.
            
        Returns
        -------
        None

        '''
        dag = complete_bipartite_dag(graphs)
        n = len(dag)
        colors= {i: random.choice([-1,0,1]) for i in range(n)} \
            if colored else None
        ud.UpDown.__init__(self, dag, coloring = colors, reduced = True)
        self.graphs = graphs

            
