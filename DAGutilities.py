#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:49:26 2020

@author: charlie
"""

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
    reverse = {v:[] for v in G.keys()}
    for v in G.keys():
        for u in G[v]:
            reverse[u].append(v)
    return reverse

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
    def descendantsDFS(G,v):
        visited[v] = 0
        reachable_from_source.append(v)
        for u in G[v]:
            if visited[u] == -1:
                descendantsDFS(G,u)
    #############################################c#############################
    reachable_from_source = []
    # keep track of nodes already visited: 0 if already visited and -1 otherwise.
    visited = {v:-1 for v in G.keys()}
    for v in G[source]:
        if visited[v] == -1:
            descendantsDFS(G,v)
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

def subGraph(G, nodes):
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
        adjacecny representation of a the sub graph of 'G' on 'nodes'.
        (Adjacecny lists keyed by node.)       

    '''
    sub_graph = {}
    for v in nodes:
        sub_graph[v] = []
        for u in G[v]:
            if u in nodes:
                sub_graph[v].append(u)              
    return sub_graph

def disjointUnion(G,H): #NEED TO FIX
    '''
    Parameters
    ----------
    G : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)
    H : dict
        adjacecny representation of a directed graph. (Adjacecny lists keyed 
        by node.)

    Returns
    ------
    dict 
        adjacecny representation** of a the disjoint union of 'G' 
        and 'H'. (Adjacecny lists keyed by node.)
    
    ** Relabels nodes to consecutive nonnegative integers 
    starting from 0.

    '''
    disjoint_union = {}
    # Relabelling for G.
    G_relabel = {}
    G_new_label = 0
    for v in G.keys():
        G_relabel[v] = G_new_label
        G_new_label += 1
    # Relabelling for H.
    H_relabel = {}
    H_new_label = G_new_label
    for v in H.keys():
        H_relabel[v] = H_new_label
        H_new_label += 1
    # Construct disjoint union
    for v in G.keys():
        disjoint_union[G_relabel[v]] = []
        for u in G[v]:
            disjoint_union[G_relabel[v]].append(G_relabel[u])
    for v in H.keys():
        disjoint_union[H_relabel[v]] = []
        for u in H[v]:
            disjoint_union[H_relabel[v]].append(H_relabel[u])         
    return disjoint_union

def topologicalSort(G, reverse = False):
    ''' Returns a topological ordering of the nodes in the directed acyclic 
    graph G.

    Parameters
    ----------
    G : dict
        adjacecny representation of an acyclic irected graph. (Adjacecny lists 
        keyed by node.)
    reverse : bool, optional
        if True, the reverse of the topological ordering will be returned. 
        The default is False.

    Returns
    -------
    list
        a topological ordering of the nodes in 'G'.
    
    Reference: 
        Introduction to Algotithms: Chapter 22, Thomas H. Cormen, 
        Charles E. Leiserson, Ronald L. Rivest, Cliffors Stein, 3rd Edition, 
        MIT Press, 2009.
        
    ** If 'G' is not acyclic an assewrtion error will be raised.

    '''
    ############# RECURSIVE HELPER ###########################################
    def topologicalSortDFS(G,v):
        discovery[v] = 0
        for u in G[v]:
            assert discovery[u] != 0, 'There is a cycle present'
            if discovery[u] == -1:
                topologicalSortDFS(G,u)  
        # Done with v, add it to ordering.
        discovery[v] = 1
        linear_order.append(v)
    ##########################################################################
    linear_order = []
    # Mark each vertex before discovery with -1, and  1 when we've finished 
    # exploring its adjacency list.
    discovery = {v:-1 for v in G.keys()}
    # Loop over all undiscovered nodes and place them in linear order once 
    # they're finished.
    for v in G.keys():
        if discovery[v] == -1:
            topologicalSortDFS(G,v)
    # Since we were appending finished nodes to the list we need to reverse.
    if not reverse:
        linear_order.reverse()
    return linear_order

def isAcyclic(G):
    ''' Returns true if teh directed graph G is acyclic and false otherwise.
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
    try:
        topologicalSort(G)
    except:
        return False
    else:
        return True

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
    
    References:
        - https://en.wikipedia.org/wiki/Transitive_closure
        - https://networkx.github.io/documentation/networkx-1.11/_modules/networkx/algorithms/dag.html#transitive_closure

    '''
    transitive_closure = {}
    # Loop over all nodees of G, adding all descendants to adjacency list. 
    for v in G.keys():
        transitive_closure[v] = descendants(G,v)
    return transitive_closure

def transitiveReduction(G):
    ''' Returns the transitive reduction of the directed acyclic graph 'G'.
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
        
    ** If 'G' is not acyclic an assertion error will be raised.
        
    References:
        - https://en.wikipedia.org/wiki/Transitive_reduction
        - https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/dag.html#transitive_reduction
    
    ''' 
    assert isAcyclic(G), 'There is a cycle present'
                       
    transitive_reduction = {}
    # Store descendants of each node as they are needed so we dont compute 
    # more times than necessary.
    G_descendants = {}
    # loop over nodes in G to compute adjacency list in transitive reduction
    for v in G.keys():
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
    ''' Returns the length of the longets path in the directed acyclic graph 'G'.
    Parameters
    ----------
    G : dict
        adjacecny representation of an acyclic directed graph. (Adjacecny lists 
        keyed by node.)

    Returns
    ------- 
    int
        length of the longest path in 'G'.
        
    ** If 'G' is not acyclic, a type error will be raised.
        
    Reference:
        https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs_and_critical_paths

    '''
    # Store maximum path length out of each node
    max_out_lengths = {v:0 for v in G.keys()}
    try:
        reverse_linear_order = topologicalSort(G, reverse = True)
    except:
        raise TypeError('G is not acyclic.')
    # Loop over nodes in reverse topological order, updating maximum length of 
    # outgoing path as we gop
    for v in reverse_linear_order:
        max_out_v = 0
        for u in G[v]:
            if max_out_lengths[u] + 1 > max_out_v:
               max_out_v = max_out_lengths[u] + 1
        max_out_lengths[v] = max_out_v
    longest_path_length = max(max_out_lengths.values())  
    return longest_path_length

    