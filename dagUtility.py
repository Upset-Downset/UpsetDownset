#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithms for diredted (acyclic) graphs provided as utility for the UpDown class.
(As noted in their docstrings, a number of the algorithms require acyclicity.
As this condition is checked via the isAcyclic() function upon instantiation of 
an upDown object we don't check for it here.)

@author: Charlie 
"""
import random

def subgraph(G, nodes):
    ''' Returns the subgraph of the directed graph 'G' on 'nodes'.
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    nodes : iterable
        nodes of 'G'

    Returns
    -------
    dict
        adjacecny representation of a the subgraph of 'G' on 'nodes'.
        (Adjacecny lists keyed by node.)       

    '''
    sub_graph = {}
    for v in nodes:
        sub_graph[v] = []
        for u in G[v]:
            if u in nodes:
                sub_graph[v].append(u)              
    return sub_graph

def reverse(G):
    ''' Returns the reverse of the (colored) directed graph 'G'.
    Parameters
    ----------
    G : dict 
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)

    Returns
    -------
    dict
        adjacecny representation of the reverse of 'G'. (Adjacecny lists keyed 
        by node.)
        
    '''
    reverse = {v:[] for v in G}
    for v in G:
        for u in G[v]:
            reverse[u].append(v)
    return reverse

def nodeColoring(G, colored = None):
    ''' Returns an all green or random blue-green-red coloring on the vertices 
    of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    coloring : string, optional
        If 'random', a random coloring will be provided. the default is an all 
        green coloring
        
    Returns
    -------
    dict
        colors (1 (resp. 0,-1) represent blue (resp. green, red)) keyed by node.

    '''
    colors = [-1,0,1]
    if colored == 'random':
        coloring = {v:random.choice(colors) for v in G}
    else:
        coloring = {v:0 for v in G}
    return coloring

def numberOfEdges(G):
    ''' Return the number of edges in the directed graph 'G'.

    Parameters
    ----------
    G : dict 
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)

    Returns
    -------
    Int
        the number of edges in 'G'.

    '''
    number_of_edges = sum(len(G[v]) for v in G)
    return number_of_edges

def inDegree(G, v):
    ''' Returns the numbner of incoming edges of the node 'v' in the directed 
    graph 'G'.
    
    Parameters
    ----------
    G : dict 
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    v : node of 'G'

    Returns
    -------
    Int
        the numbner of incoming edges of 'v' in 'G'.

    '''
    return len(G[v])

def outDegree(G, v):
    ''' Returns the numbner of outgoing edges of the node 'v' in the directed 
    graph 'G'.
    
    Parameters
    ----------
    G : dict 
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    v : node of 'G'

    Returns
    -------
    Int
        the numbner of outgoing edges of 'v' in 'G'.

    '''
    rev = reverse(G)
    return len(rev[v])

def descendants(G, source):
    ''' Returns all nodes reachable from 'source' in the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    source : node of 'G'. 

    Returns
    -------
    list
        all nodes reachbale from 'source' in 'G'.
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.    
    '''
    ############# RECURSIVE HELPER ###########################################
    def dfsVisit(G,v):
        visited[v] = 0
        reachable_from_source.append(v)
        for u in G[v]:
            if visited[u] == -1:
                dfsVisit(G,u)
    #############################################c#############################
    reachable_from_source = []
    # Keep track of nodes already visited by marking with 0 if already visited
    # and -1 otherwise.
    visited = {v:-1 for v in G}
    for v in G[source]:
        if visited[v] == -1:
            dfsVisit(G,v)
    return reachable_from_source
   
def ancestors(G, source):
    ''' Returns all nodes having a path to 'source' in the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    source : node of 'G'.

    Returns
    -------
    list
        all nodes having a path to 'source' in 'G'.

    '''
    rev = reverse(G)
    has_path_to_source = descendants(rev, source)
    return has_path_to_source

def isAcyclic(G):
    ''' Returns true if the directed graph G is acyclic and false otherwise.
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)

    Returns
    -------
    bool
        True if 'G' is acyclic and False otherwise.
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.

    '''
   ############# RECURSIVE HELPER ###########################################
    def dfsVisit(G,v):
        discovery[v] = 0
        for u in G[v]:
            # If we encounter a node again before we've finished exploring it's 
            # adjacecny list there must be a cycle!
            if discovery[u] == 0:
                return False
            if discovery[u] == -1:
                if not dfsVisit(G,u):
                    return False
        # Done sxploring v.
        discovery[v] = 1
        return True
    ##########################################################################
    # Mark each vertex before discovery with -1, with 0 once discovered and 
    # 1 when we've finished exploring its adjacency list.
    discovery = {v:-1 for v in G}
    # Loop over all undiscovered nodes and check for cycles.
    for v in G.keys():
        if discovery[v] == -1:
            if not dfsVisit(G,v):
                return False
    return True

def topologicalSort(G, reverse = False):
    ''' Returns a topological ordering of the nodes in the directed acyclic**
    graph G.

    Parameters
    ----------
    G : dict
        adjacecny representation of an acyclic directed graph. (Adjacecny lists 
        keyed by node.)
    reverse : bool, optional
        if True, the reverse of the topological ordering will be returned. 
        The default is False.

    Returns
    -------
    list
        a topological ordering of the nodes in 'G'.
        
    ** It is assumed that 'G' is acyclic, we do not check.
    
    Reference:
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.

    '''
    ############# RECURSIVE HELPER ###########################################
    def dfsVisit(G,v):
        discovery[v] = 0
        for u in G[v]:
            if discovery[u] == -1:
                dfsVisit(G,u)  
        # Done with v, add it to ordering.
        discovery[v] = 1
        linear_order.append(v)
    ##########################################################################
    linear_order = []
    # Mark each vertex before discovery with -1, with 0 once discovered and 
    # 1 when we've finished exploring its adjacency list.
    discovery = {v:-1 for v in G}
    # Loop over all undiscovered nodes and place them in linear order once 
    # they're finished.
    for v in G.keys():
        if discovery[v] == -1:
            dfsVisit(G,v)
    # Since we were appending finished nodes to the list we need to reverse.
    if not reverse:
        linear_order.reverse()
    return linear_order

def transitiveClosure(G):
    ''' Returns the transitive closure of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)

    Returns
    -------
    dict
        adjacecny representation of the transitive closure of 'G'. (Adjacecny 
        lists keyed by node.)
        
    Reference:
        https://en.wikipedia.org/wiki/Transitive_closure
         
    '''
    transitive_closure = {}
    # Loop over all nodes of G, adding all descendants to adjacency list in 
    # transitive closure. 
    for v in G.keys():
        transitive_closure[v] = descendants(G,v)
    return transitive_closure

def transitiveReduction(G):
    ''' Returns the transitive reduction of the directed acyclic** graph 'G'.
    Parameters.
    ----------
    G : dict
        adjacecny representation of an acyclic directed graph. (Adjacecny lists 
        keyed by node.)

    Returns
    -------
    dict
        adjacecny representation of the transitive reduction of 'G'. (Adjacecny 
        lists keyed by node.
        
    ** It is assumed that 'G' is acyclic, we do not check.
    
    References:
        - https://en.wikipedia.org/wiki/Transitive_reduction
        - https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/dag.html
    '''                 
    transitive_reduction = {}
    # Store descendants of each node as they are needed so we don't compute 
    # more times than necessary.
    G_descendants = {}
    # loop over nodes in G to compute adjacency list in transitive reduction
    for v in G:
        # Store nodes adjacent to v in transitive reduction.
        tr_adj_v = set(G[v])
        # loop over nodes adjacent to v in G removing their descendants from 
        # v's adjaceny list in transitive reduction
        for u in G[v]:
            # If u has already been removed from v's adjaceny list in 
            # transitive reduction, then so have u's descendants in G.
            if u in tr_adj_v:
                # if necessary, compute u's descendants in G.
                if u not in G_descendants:
                    G_descendants[u] = set(descendants(G,u))
                # Remove u's descendants in G from v's adjacency list in 
                # transitive reduction
                tr_adj_v -= G_descendants[u]
        # update v's adjaceny list in transitive reduction
        transitive_reduction[v] = list(tr_adj_v)
    return transitive_reduction

def longestPathLength(G):
    ''' Returns the length of the longest path in the directed acyclic** graph 'G'.
    Parameters
    ----------
    G : dict
        adjacecny representation of an acyclic directed graph. (Adjacecny lists 
        keyed by node.)

    Returns
    ------- 
    int
        length of the longest path in 'G'.
        
    ** It is assumed that 'G' is acyclic, we do not check.
    
    Reference: 
        - https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs_and_critical_paths

    '''
    # Store maximum path length out of each node
    max_out_lengths = {v:0 for v in G}
    reverse_linear_order = topologicalSort(G, reverse = True)
    # Loop over nodes in reverse topological order, updating maximum length of 
    # outgoing path as we go.
    for v in reverse_linear_order:
        max_out_v = 0
        for u in G[v]:
            if max_out_lengths[u] + 1 > max_out_v:
               max_out_v = max_out_lengths[u] + 1
        max_out_lengths[v] = max_out_v
    longest_path_length = max(max_out_lengths.values())  
    return longest_path_length

    