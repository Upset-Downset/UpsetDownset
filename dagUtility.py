#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Charles Petersen and Jamison Barsotti
"""

import copy
import random

def relabel(G, relabel_map):
    ''' Relabel nodes of the directed graph 'G' according to 'relabel_map'. 
    
    Parameters
    
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    relabel_map : dict
        new node labels keyed by old node labels. 
    Returns
    -------
    graph: dict
        adjacecny representation of the directed graph 'G' with nodes relabelled by 'relabel_map'.
    '''
    graph = {}
    for v in G:
        graph[relabel_map[v]] = []
        for u in G[v]:
            graph[relabel_map[v]].append(relabel_map[u])
    return graph
 
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
    ''' Returns the reverse of the directed graph 'G'.
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

def edge_list(G):
    '''Returns list of edges of the directed graph 'G'.
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed acyclic graph. (Adjacecny lists 
        keyed by node.)

    Returns
    -------
    list
        lists of ordered-pairs denoting the edges of G.

    '''
    #a list to store the edges
    edges = []
    #iterate over the nodes, then iterate over the the covers of that node.
    #collect all the pairs.
    for v in G:
        for u in G[v]:
            edges.append((u,v))   
    return edges

def number_of_edges(G):
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
    return sum(len(G[v]) for v in G)

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
    def dfs_visit(G,v):
        visited[v] = 0
        reachable_from_source.append(v)
        for u in G[v]:
            if visited[u] == -1:
                dfs_visit(G,u)
    ##########################################################################
    reachable_from_source = []
    # Keep track of nodes already visited by marking with 0 if already visited
    # and -1 otherwise.
    visited = {v:-1 for v in G}
    for v in G[source]:
        if visited[v] == -1:
            dfs_visit(G,v)
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

def is_acyclic(G):
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
    def dfs_visit(G,v):
        discovery[v] = 0
        for u in G[v]:
            # If we encounter a node again before we've finished exploring it's 
            # adjacecny list there must be a cycle!
            if discovery[u] == 0:
                return False
            if discovery[u] == -1:
                if not dfs_visit(G,u):
                    return False
        # Done sxploring v.
        discovery[v] = 1
        return True
    ##########################################################################
    # Mark each vertex before discovery with -1, with 0 once discovered and 
    # 1 when we've finished exploring its adjacency list.
    discovery = {v:-1 for v in G}
    # Loop over all undiscovered nodes and check for cycles.
    for v in G:
        if discovery[v] == -1:
            if not dfs_visit(G,v):
                return False
    return True

def topological_sort(G, reverse = False):
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
    def dfs_visit(G,v):
        discovery[v] = 0
        for u in G[v]:
            if discovery[u] == -1:
                dfs_visit(G,u)  
        # Done with v, add it to ordering.
        discovery[v] = 1
        linear_order.append(v)
    ##########################################################################
    linear_order = []
    # Mark each vertex before discovery with -1, with 0 once discovered and 
    # 1 when we've finished exploring its adjacency list.
    discovery = {v:-1 for v in G}
    # Loop over all undiscovered nodes placing in order as we go.
    for v in G:
        if discovery[v] == -1:
            dfs_visit(G,v)
    # Since we were appending finished nodes to the list we need to reverse.
    if not reverse:
        linear_order.reverse()
    return linear_order

def transitive_closure(G):
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
    for v in G:
        transitive_closure[v] = descendants(G,v)
    return transitive_closure

def transitive_reduction(G):
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
        - https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/dag.html
        - https://en.wikipedia.org/wiki/Transitive_reduction
        
    *** All credit to the authors of the algorithm found in networkx's dag 
        code!
    '''                 
    transitive_reduction = {}
    # Store descendants of each node in G in a dict as they are needed so we 
    # don't compute more times than necessary.
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
                # If necessary, compute u's descendants in G.
                if u not in G_descendants:
                    G_descendants[u] = set(descendants(G,u))
                # Remove u's descendants in G from v's adjacency list in 
                # transitive reduction
                tr_adj_v -= G_descendants[u]
        # update v's adjaceny list in transitive reduction
        transitive_reduction[v] = list(tr_adj_v)
    return transitive_reduction

def longest_path_lengths(G, direction = 'outgoing'):
    ''' Returns the length of the longest path 'outgoing' (optionally, incoming) 
    each node in the directed acyclic** graph 'G' 
    
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed acyclic graph. (Adjacecny lists 
        keyed by node.)
    direction : str, optional
         If 'outgoing', the length of the longest path starting at each node will
         length will be omputed. If 'incoming', the length of the longest path 
         ending at each node will length will be computed.

    Returns
    ------- 
    dict
        lengths of the longest outgoing (or incoming) paths in 'G' 'keyed by 
        node .
        
    ** It is assumed that 'G' is acyclic, we do not check.
    
    Reference: 
        - https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs_and_critical_paths

    '''
    # Store maximum path length computed for each node
    max_path_lengths = {v:0 for v in G}
    # If we want incoming paths, then reverse the graph.
    if direction == 'incoming':
        G = reverse(G)
    reverse_linear_order = topological_sort(G, reverse = True)
    # Loop over nodes in reverse topological order, updating maximum length of 
    # each path as we go.
    for v in reverse_linear_order:
        MAX = 0
        for u in G[v]:
            if max_path_lengths[u] + 1 > MAX:
               MAX = max_path_lengths[u] + 1
        max_path_lengths[v] = MAX  
    return max_path_lengths

def connected_components(G):
    ''' Returns the nodes in each component of the undirected 
    graph underlying the directed graph 'G"."

    Parameters
    ----------
    G : dict
        adjacecny representation of a directed acyclic graph. (Adjacecny lists 
        keyed by node.)

    Returns
    -------
    list
        set of nodes in each connected compnent of the undirected graph 
        underlying 'G'.
    '''
    ############# RECURSIVE HELPER ###########################################
    def dfs_visit(G,v):
        discovery[v] = 0
        for u in undirected[v]:
            if discovery[u] == -1:
                v_component.add(u)
                dfs_visit(undirected,u)  
        # Done with v.
        discovery[v] = 1
    ########################################################################## 
    # initialize an undirected version of G.    
    rev = reverse(G)
    undirected = copy.deepcopy(G)
    for v in undirected:
        undirected[v].extend(rev[v])
    # store nodes in G in a set to be updated as we discover each component.
    nodes = set(G)
    # to store the components
    components = []
    # Mark each vertex before discovery with -1, with 0 once discovered and 
    # 1 when we've finished exploring its adjacency list.
    discovery = {v:-1 for v in G}
    # choose a node from G.
    v = nodes.pop()
    # dfs to find all components of G, updated nodes set as we go.
    while True:
        v_component = {v}
        dfs_visit(undirected,v)
        components.append(v_component)
        nodes = nodes - v_component
        if nodes:
            v = nodes.pop()
        else: 
            break
    return components

    